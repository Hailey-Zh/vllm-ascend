#!/usr/bin/env python3
"""
验证 compressor 第一段 matmul 的精度问题。

背景：训练参考用小算子对齐，做的是 `x.to(fp32) @ w.to(fp32)`。
算子里 (arch32: op_kernel/arch32/compressor_block_cube_perf.h;
        arch35: op_kernel/arch35/compressor_block_cube.h) Cube 做的是:
    - A/B 进 L0 是 bf16/fp16 (X_T)
    - C (L0C) 是 fp32, Mmad 把 bf16×bf16 的乘积按 fp32 累加 (MM1_OUT_T = float)
    - K 方向按 K_L0_BASE=128 分块累加

本脚本验证一个论断:
    "bf16×bf16 精确乘积 + fp32 累加"  ==  "cast 到 fp32 再矩阵乘"
两者数学等价, 唯一差别是累加顺序 (~1e-7 相对量级)。

关键事实: bf16(尾数8位) × bf16 的精确积最多约 16 位尾数, fp32(24位)无损放得下;
fp16 同理(11位×→22位<24)。所以 cast 到 fp32 不会带来任何新信息。

纯 CPU 运行:  python verify_matmul_precision.py
"""

import torch


def run(dtype: torch.dtype, T: int = 256, H: int = 4096, N: int = 512, k_block: int = 128):
    """
    形状参照 README: x[T, H], wkv[N=coff*D, H], 输出 [T, N] = x @ wkv^T。
    dtype: torch.bfloat16 或 torch.float16 —— 对应算子输入 X_T。
    """
    torch.manual_seed(0)
    # 这就是算子真正拿到的输入: 已经是 bf16/fp16 精度
    x_lp = torch.empty(T, H, dtype=dtype).uniform_(-1.0, 1.0)
    w_lp = torch.empty(N, H, dtype=dtype).uniform_(-1.0, 1.0)

    xf = x_lp.float()  # 无损升位: 数值不变, 只是补零尾数
    wf = w_lp.float()

    # (a) 训练参考: cast 到 fp32 后整体矩阵乘
    ref = xf @ wf.t()

    # (b) 模拟 Cube: 沿 K 按 128 分块, 每块 fp32 累加 —— 复刻 kernel 的累加顺序。
    #     因为 bf16/fp16 × 同类的精确积在 fp32 中无损, 用 fp32 算每块即等价于硬件 Cube。
    cube_sim = torch.zeros(T, N, dtype=torch.float32)
    for k in range(0, H, k_block):
        cube_sim += xf[:, k:k + k_block] @ wf[:, k:k + k_block].t()

    # (c) 对照组: 如果把 matmul 输出 round 回 bf16/fp16 才会有的误差。
    #     注意算子并没有这么做 —— MM1_OUT_T = float, 中间结果保持 fp32。
    out_rounded = ref.to(dtype).float()

    scale = ref.abs().max().clamp_min(1e-12)
    diff_cube = (ref - cube_sim).abs().max()
    diff_round = (ref - out_rounded).abs().max()

    print(f"\n===== dtype = {dtype} (T={T}, H={H}, N={N}, K_block={k_block}) =====")
    print(f"[参考 vs Cube(分块fp32累加)]  max|Δ| = {diff_cube.item():.3e}   "
          f"rel = {(diff_cube / scale).item():.3e}   <- 这就是算子 matmul 实际差异(仅累加顺序)")
    print(f"[参考 vs 输出round成{str(dtype).split('.')[-1]}]  max|Δ| = {diff_round.item():.3e}   "
          f"rel = {(diff_round / scale).item():.3e}   <- 算子并不这么做, 仅作对照")


def perturbation_hint():
    print("""
------------------------------------------------------------------
判定方法 (定位 gap 在不在 matmul):
  在你完整的参考实现 compressor_ref 上做扰动实验, 只改 matmul 操作数精度:

    out_fp32 = compressor_ref(x.float(),            w.float(),            ...)
    out_lp   = compressor_ref(x.bfloat16().float(), w.bfloat16().float(), ...)
    mm_gap   = (out_fp32 - out_lp).abs().max()

  mm_gap = matmul 用低精度输入 对【最终输出】影响的上限。
    - mm_gap ≈ 实际(算子 vs golden) gap  -> matmul 输入精度是根因
    - mm_gap << 实际 gap                  -> 差异在下游 softmax/norm, 别动 Cube
------------------------------------------------------------------""")


if __name__ == "__main__":
    run(torch.bfloat16)
    run(torch.float16)
    perturbation_hint()
