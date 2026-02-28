import argparse
import torch
import torch.nn.functional as F
import os
import sys
import math

# =========================================================================
# 【核心修复】定义一个普通的 Attention 函数，用来替换 PyTorch 2.0 的高级算子
#  这个函数只使用了基础的 MatMul, Transpose, Softmax，所有推理引擎都支持
# =========================================================================
def manual_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    # 1. 计算 Q * K^T
    # attn_weight: [Batch, Heads, L, S]
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    
    # 2. 处理 Mask (如果有)
    if is_causal:
        # 处理因果掩码 (通常用于 GPT 类模型，这里 AVTrack 可能用不到，但为了兼容写上)
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_weight.masked_fill_(temp_mask.logical_not(), float("-inf"))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_weight += attn_mask

    # 3. Softmax
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    # 4. Dropout (导出时通常为 0，忽略)
    
    # 5. Output: Score * V
    return attn_weight @ value

# 【注入补丁】强行覆盖 PyTorch 的官方实现
# 这样在导出时，PyTorch 就会调用上面这个函数，而不是底层的 C++ 加速算子
torch.nn.functional.scaled_dot_product_attention = manual_scaled_dot_product_attention
# =========================================================================

import argparse
import torch
import os
import sys
import importlib
import math # 引入 math
# 将当前目录加入 python path，确保能 import lib
sys.path.append(os.getcwd())

# 引入项目特定的构建函数
from lib.models.avtrack import build_avtrack
from lib.config.avtrack.config import cfg, update_config_from_file

# =========================================================================
# 1. 定义包装类 (Wrapper)
#    作用：
#    1. 自动生成 dummy_anno，解决 forward 参数报错问题
#    2. 自动拆包 out 字典，只返回 score, size, offset
# =========================================================================
class AVTrackExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 设为 eval 模式，这一步至关重要！
        # 它会告诉模型：“我在推理，不要计算 Loss，不要做 Dropout”
        self.model.eval()

    def forward(self, template, search):
        # 构造假的标注数据 (Batch_Size, 4)
        # 因为 eval() 模式下模型其实不用这些数据，但函数签名要求必须传
        B = template.shape[0]
        dummy_anno = torch.zeros((B, 4), device=template.device)

        # 调用原始模型
        # 注意参数顺序：template, search, template_anno, search_anno
        out = self.model(template, search, dummy_anno, dummy_anno)

        # === 核心拆分逻辑 ===
        # 从字典中提取我们要的“三剑客”
        score = out['score_map']
        size = out['size_map']
        offset = out['offset_map']

        # 返回 Tuple (ONNX 要求)
        return score, size, offset


def parse_args():
    parser = argparse.ArgumentParser(description='AVTrack ONNX Export')
    # 配置文件路径 (请根据你实际训练的 yaml 修改)
    parser.add_argument('--config', type=str, default='experiments/avtrack/baseline.yaml', 
                        help='yaml configure file name')
    # 权重文件路径 (请修改为你训练好的 pth)
    parser.add_argument('--checkpoint', type=str, default='snapshot/avtrack_ep0020.pth',
                        help='checkpoint file name')
    # 导出文件名
    parser.add_argument('--output', type=str, default='avtrack.onnx')
    return parser.parse_args()


def main():
    args = parse_args()

    # =========================================================================
    # 2. 加载配置和模型
    # =========================================================================
    print(f"Loading config from: {args.config}")
    update_config_from_file(args.config)

    print("Building AVTrack model...")
    # training=False 很重要，虽然我们后面还会 model.eval()，但这里最好也设一下
    model_raw = build_avtrack(cfg, training=False)

    # 加载权重
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 处理权重字典 Key 不匹配的问题 (常见的 'net' 包裹)
    if 'net' in checkpoint:
        model_raw.load_state_dict(checkpoint['net'], strict=False)
    else:
        model_raw.load_state_dict(checkpoint, strict=False)
    
    # 转到 GPU (如果要在 CPU 导出，改 .cpu())
    model_raw.cuda()

    # =========================================================================
    # 3. 包装模型
    # =========================================================================
    model = AVTrackExport(model_raw)

    # =========================================================================
    # 4. 构造虚拟输入 (Dummy Input)
    #    形状必须和训练/推理时完全一致！
    #    通常 Template=128, Search=256 (具体看你的 yaml DATA.TEMPLATE.SIZE)
    # =========================================================================
    # 尝试从配置读取尺寸，读不到就用默认值
    t_sz = cfg.DATA.TEMPLATE.SIZE if hasattr(cfg.DATA, 'TEMPLATE') else 128
    s_sz = cfg.DATA.SEARCH.SIZE if hasattr(cfg.DATA, 'SEARCH') else 256
    
    print(f"Dummy Input Size -> Template: {t_sz}, Search: {s_sz}")
    
    template = torch.randn(1, 3, t_sz, t_sz).cuda()
    search = torch.randn(1, 3, s_sz, s_sz).cuda()

    # =========================================================================
    # 5. 执行导出
    # =========================================================================
    print(f"Exporting to {args.output} ...")
    
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        torch.onnx.export(
            model,                  # 包装后的模型
            (template, search),     # 输入元组
            args.output,            # 输出路径
            opset_version=11,       # 算子集版本
            do_constant_folding=True,
            input_names=['template', 'search'],
            output_names=['score_map', 'size_map', 'offset_map'],
            # dynamic_axes=... 
        )

    print("✅ Export Success!")
    print(f"Run: netron {args.output} to visualize the model.")

    # =========================================================================
    # 6. (可选) 验证导出模型的正确性
    # =========================================================================
    try:
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX Model Check Passed!")
    except Exception as e:
        print(f"❌ ONNX Model Check Failed: {e}")


if __name__ == '__main__':
    main()