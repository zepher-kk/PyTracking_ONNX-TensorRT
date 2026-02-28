import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os
import gc

# ================= Configuration =================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

class OSTrackEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        print(f"Loading Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = self.engine.get_tensor_shape(name)
            if shape[0] == -1: shape = (1,) + shape[1:]
            size = trt.volume(shape) * np.dtype(np.float32).itemsize
            allocation = cuda.mem_alloc(size)
            self.allocations.append(allocation)
            self.context.set_tensor_address(name, int(allocation))
            binding = {'index': i, 'name': name, 'host': np.zeros(shape, dtype=np.float32), 'device': allocation, 'shape': shape}
            if is_input: self.inputs.append(binding)
            else: self.outputs.append(binding)
        self.input_dict = {inp['name']: inp for inp in self.inputs}
        self.output_dict = {out['name']: out for out in self.outputs}

    def infer(self, template_img, search_img):
        z = self._preprocess(template_img, 128)
        x = self._preprocess(search_img, 256)
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

def crop_hwc(image, bbox, out_sz, factor=4.0):
    x, y, w, h = bbox
    cx, cy = x+w/2, y+h/2
    area = (w + 0.5*(w+h)) * (h + 0.5*(w+h))
    
    # 保持 factor/2.0 (2倍视野)，因为这是您测试下来效果最好的
    crop_sz = np.ceil(np.sqrt(area) * (factor/2.0))
    
    h_img, w_img, _ = image.shape
    crop_sz = min(crop_sz, max(h_img, w_img)*1.5)
    if crop_sz < 10: crop_sz = 10
    x1, y1 = int(round(cx-crop_sz/2)), int(round(cy-crop_sz/2))
    x2, y2 = x1+int(round(crop_sz)), y1+int(round(crop_sz))
    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad, y2_pad = max(0, x2-w_img), max(0, y2-h_img)
    im_crop = image[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad]
    if x1_pad or x2_pad or y1_pad or y2_pad:
        try: im_crop = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        except: im_crop = cv2.resize(image, (int(out_sz), int(out_sz)))
    return im_crop, crop_sz

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess(score_map, size_map, offset_map, crop_sz, search_size=256):
    # 还原为 (C, H, W) 形状，feat_sz = 16
    score = score_map.reshape(16, 16)
    size  = size_map.reshape(2, 16, 16)
    offset = offset_map.reshape(2, 16, 16)
    
    # 1. 汉宁窗惩罚 (Hanning Window)
    if not hasattr(postprocess, 'hann'):
        hann = np.hanning(16)
        postprocess.hann = np.outer(hann, hann)
    
    # 此时 score 已由 ONNX 内部的 Sigmoid 处理过，直接乘窗即可
    score_final = score * postprocess.hann
    
    # 2. 定位最大响应点 (相当于 torch.max)
    idx = np.argmax(score_final)
    y_idx, x_idx = np.unravel_index(idx, (16, 16))
    conf_score = score[y_idx, x_idx] # 使用未加窗的原始置信度
    
    # 3. 提取 Size 和 Offset (严格遵循源码通道顺序)
    w_raw = size[0, y_idx, x_idx]
    h_raw = size[1, y_idx, x_idx]
    
    off_x = offset[0, y_idx, x_idx]
    off_y = offset[1, y_idx, x_idx]
    
    # 4. 坐标解码 (一比一复刻 cal_bbox)
    feat_sz = 16.0
    # 得到相对于搜索区域 [0, 1] 的相对坐标
    cx_rel = (x_idx + off_x) / feat_sz
    cy_rel = (y_idx + off_y) / feat_sz
    
    # size_map 在 ONNX 里已经过了 Sigmoid，本身就是 [0, 1] 的比例
    w_rel = w_raw
    h_rel = h_raw
    
    # 5. 映射回 256x256 像素尺度
    cx_pixel = cx_rel * search_size
    cy_pixel = cy_rel * search_size
    w_pixel = w_rel * search_size
    h_pixel = h_rel * search_size
    
    # 6. 映射回原图偏移量 (Map back to Original Image)
    # 计算当前预测中心点，相对于 Search Region 中心的偏差
    scale = crop_sz / search_size
    delta_x = (cx_pixel - search_size / 2.0) * scale
    delta_y = (cy_pixel - search_size / 2.0) * scale
    w_real = w_pixel * scale
    h_real = h_pixel * scale
    
    # (可选) 返回热力图可视化
    score_vis = (score - score.min()) / (score.max() - score.min() + 1e-5) * 255
    score_vis = cv2.applyColorMap(score_vis.astype(np.uint8), cv2.COLORMAP_JET)
    score_vis = cv2.resize(score_vis, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    return delta_x, delta_y, w_real, h_real, conf_score, score_vis

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--engine', type=str, default='/mnt/d/learn/track/OSTrack/ostrack_vitb_256_fp16.engine')
    args = parser.parse_args()

    tracker = OSTrackEngine(args.engine)
    cap = cv2.VideoCapture(args.video)
    w_vid, h_vid = int(cap.get(3)), int(cap.get(4))
    
    ret, frame = cap.read()
    if not ret: return
    
    print("=== 请框选目标 (左上角 -> 右下角) ===")
    bbox = cv2.selectROI('Track', frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow('Track')
    
    if bbox[2] == 0: return
    
    # 【修复1】严格保持 state 为 [x_topleft, y_topleft, w, h]
    state = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    
    template_img, _ = crop_hwc(frame, state, 128, factor=2.0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 此时 state 传进去的是标准的 [x, y, w, h]，crop_hwc 计算中心就对了
        search_img, actual_crop_sz = crop_hwc(frame, state, 256, factor=4.0)
        score, size, offset = tracker.infer(template_img, search_img)
        
        dx, dy, w, h, conf, score_vis = postprocess(score, size, offset, actual_crop_sz)
        
        # 【修复2】正确的坐标更新逻辑
        # 1. 计算当前(旧的)中心点
        cx = state[0] + state[2] / 2.0
        cy = state[1] + state[3] / 2.0
        
        # 2. 加上模型预测的偏差 (dx, dy 是相对于旧中心的增量)
        cx += dx
        cy += dy
        
        # 3. 平滑更新宽高
        state[2] = state[2] * 0.6 + w * 0.4
        state[3] = state[3] * 0.6 + h * 0.4
        
        # 4. 根据新中心和新宽高，算回左上角，存回 state
        state[0] = cx - state[2] / 2.0
        state[1] = cy - state[3] / 2.0
        
        # 防止出界
        state[2] = min(state[2], w_vid)
        state[3] = min(state[3], h_vid)
        
        # 绘图 (现在 state[0] 就是 x1，可以直接取)
        x1 = int(round(state[0]))
        y1 = int(round(state[1]))
        x2 = int(round(state[0] + state[2]))
        y2 = int(round(state[1] + state[3]))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {conf:.2f}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Track', frame)
        cv2.imshow('Heatmap', score_vis)
        
        if cv2.waitKey(30) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()