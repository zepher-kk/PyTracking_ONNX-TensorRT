import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# ================= 基础配置 =================
TEMPLATE_SIZE = 128
SEARCH_SIZE = 256
SEARCH_FACTOR = 4.0   # 根据你实际测试的效果，通常 Search Region 是目标的 4.0 倍
TEMPLATE_FACTOR = 2.0 # Template 通常是 2.0 倍

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        print(f"Loading AVTrack Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.inputs, self.outputs, self.allocations = [], [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            if shape[0] == -1: shape = (1,) + shape[1:]
            size = trt.volume(shape) * np.dtype(np.float32).itemsize
            allocation = cuda.mem_alloc(size)
            self.allocations.append(allocation)
            self.context.set_tensor_address(name, int(allocation))
            binding = {'index': i, 'name': name, 'host': np.zeros(shape, dtype=np.float32), 'device': allocation, 'shape': shape}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: self.inputs.append(binding)
            else: self.outputs.append(binding)
        self.input_dict = {inp['name']: inp for inp in self.inputs}
        self.output_dict = {out['name']: out for out in self.outputs}

    def infer(self, template, search):
        z = self._preprocess(template, TEMPLATE_SIZE)
        x = self._preprocess(search, SEARCH_SIZE)
        
        # 兼容不同的 ONNX 导出节点名
        z_key = 'z' if 'z' in self.input_dict else 'template'
        x_key = 'x' if 'x' in self.input_dict else 'search'
        
        np.copyto(self.input_dict[z_key]['host'], z)
        np.copyto(self.input_dict[x_key]['host'], x)
        for inp in self.inputs: cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for out in self.outputs: cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return (self.output_dict['score_map']['host'], self.output_dict['size_map']['host'], self.output_dict['offset_map']['host'])

    def _preprocess(self, img, size):
        if img.shape[0] != size or img.shape[1] != size: img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]
        img = (img - MEAN) / STD
        return np.ascontiguousarray(img)

def crop_hwc(image, bbox, out_sz, padding_factor):
    # 此处 bbox 必须是 [x_topleft, y_topleft, width, height]
    x, y, w, h = bbox
    cx, cy = x + w/2.0, y + h/2.0
    
    # 【核心修复】：严格遵守 OSTrack 官方公式 math.sqrt(w * h) * factor
    crop_sz = np.ceil(np.sqrt(w * h) * padding_factor)
    crop_sz = max(crop_sz, 10.0) # 防止极端过小
    
    x1 = int(round(cx - crop_sz / 2.0))
    y1 = int(round(cy - crop_sz / 2.0))
    x2 = int(x1 + round(crop_sz))
    y2 = int(y1 + round(crop_sz))
    
    h_img, w_img, _ = image.shape
    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad, y2_pad = max(0, x2 - w_img), max(0, y2 - h_img)
    
    im_crop = image[y1+y1_pad : y2-y2_pad, x1+x1_pad : x2-x2_pad]
    
    if x1_pad or x2_pad or y1_pad or y2_pad:
        try: 
            im_crop = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        except: 
            im_crop = cv2.resize(image, (int(out_sz), int(out_sz)))
            
    if im_crop.shape[0] == 0 or im_crop.shape[1] == 0:
        return np.zeros((out_sz, out_sz, 3), dtype=np.uint8), crop_sz
        
    return cv2.resize(im_crop, (out_sz, out_sz)), crop_sz

def postprocess(score_map, size_map, offset_map, crop_sz, search_size=256):
    score = score_map.reshape(16, 16)
    size  = size_map.reshape(2, 16, 16)
    offset = offset_map.reshape(2, 16, 16)
    
    # 1. 汉宁窗惩罚
    if not hasattr(postprocess, 'hann'):
        hann = np.hanning(16)
        postprocess.hann = np.outer(hann, hann)
    
    score_final = score * postprocess.hann
    
    # 2. 定位最大响应点 (等同于 torch.max)
    idx = np.argmax(score_final)
    y_idx, x_idx = np.unravel_index(idx, (16, 16))
    conf_score = score[y_idx, x_idx]
    
    # 3. 提取 Size 和 Offset
    # 源码: bbox = torch.cat([... size.squeeze(-1)]) => 证明 Ch0=W, Ch1=H
    w_raw = size[0, y_idx, x_idx]
    h_raw = size[1, y_idx, x_idx]
    
    # 源码: (idx_x + offset[:, :1]) => 证明 Ch0=X, Ch1=Y，且没有 Sigmoid，直接相加！
    off_x = offset[0, y_idx, x_idx]
    off_y = offset[1, y_idx, x_idx]
    
    # 4. 还原相对坐标 [0, 1]
    feat_sz = 16.0
    cx_rel = (x_idx + off_x) / feat_sz
    cy_rel = (y_idx + off_y) / feat_sz
    w_rel = w_raw
    h_rel = h_raw
    
    # 5. 映射回 256x256 像素尺度
    cx_pixel = cx_rel * search_size
    cy_pixel = cy_rel * search_size
    w_pixel = w_rel * search_size
    h_pixel = h_rel * search_size
    
    # 6. 映射回原图偏差量
    scale = crop_sz / search_size
    delta_x = (cx_pixel - search_size / 2.0) * scale
    delta_y = (cy_pixel - search_size / 2.0) * scale
    w_real = w_pixel * scale
    h_real = h_pixel * scale
    
    # (可选) 可视化热力图
    score_vis = (score - score.min()) / (score.max() - score.min() + 1e-5) * 255
    score_vis = cv2.applyColorMap(score_vis.astype(np.uint8), cv2.COLORMAP_JET)
    score_vis = cv2.resize(score_vis, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    return delta_x, delta_y, w_real, h_real, conf_score, score_vis
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--engine', type=str, required=True) # 传入你的 AVTrack engine 路径
    args = parser.parse_args()

    tracker = TRTEngine(args.engine)
    cap = cv2.VideoCapture(args.video)
    w_vid, h_vid = int(cap.get(3)), int(cap.get(4))
    
    is_paused = False
    
    print("=== 🛠️ AVTrack Perfect TRT 🛠️ ===")
    print(" [Space]: 暂停/继续")
    print(" [R]: 重新框选")
    
    ret, frame = cap.read()
    if not ret: return
    
    bbox = cv2.selectROI('AVTrack', frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow('AVTrack')
    
    if bbox[2] == 0: return
    
    # 【核心修复】：统一 state 为 [x_topleft, y_topleft, w, h]
    state = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    template_img, _ = crop_hwc(frame, state, TEMPLATE_SIZE, padding_factor=TEMPLATE_FACTOR)
    
    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            search_img, actual_crop_sz = crop_hwc(frame, state, SEARCH_SIZE, padding_factor=SEARCH_FACTOR)

            score, size, offset = tracker.infer(template_img, search_img)
            dx, dy, w, h, conf, score_vis = postprocess(score, size, offset, actual_crop_sz)
            
            if conf > 0.15: # 置信度阈值
                # 1. 计算旧中心点
                cx = state[0] + state[2] / 2.0
                cy = state[1] + state[3] / 2.0
                
                # 2. 累加网络预测的中心点偏移量
                cx += dx
                cy += dy
                
                # 3. 尺寸平滑更新
                state[2] = state[2] * 0.8 + w * 0.2
                state[3] = state[3] * 0.8 + h * 0.2
                
                # 4. 反算回左上角坐标，保持 state 定义一致
                state[0] = cx - state[2] / 2.0
                state[1] = cy - state[3] / 2.0
            
            # 防止框出界
            state[2] = min(state[2], w_vid)
            state[3] = min(state[3], h_vid)
            
            # 绘图
            x1, y1 = int(round(state[0])), int(round(state[1]))
            x2, y2 = int(round(state[0] + state[2])), int(round(state[1] + state[3]))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf:{conf:.2f}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('AVTrack', frame)
            # cv2.imshow('Heatmap', score_vis) # 取消注释可看热力图
        
        delay = 30 if not is_paused else 0
        key = cv2.waitKey(delay)
        
        if key == ord('q'): break
        elif key == ord(' '): is_paused = not is_paused
        elif key == ord('r'):
            bbox = cv2.selectROI('AVTrack', frame, showCrosshair=True, fromCenter=False)
            if bbox[2] != 0:
                state = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                template_img, _ = crop_hwc(frame, state, TEMPLATE_SIZE, padding_factor=TEMPLATE_FACTOR)
            is_paused = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()