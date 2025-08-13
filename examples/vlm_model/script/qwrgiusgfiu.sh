#!/usr/bin/env bash

# 指定 GPU ID
GPU_ID=2
# 占用多少显存（GB）
TARGET_GB=75

# 显式绑定到指定 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

python - <<EOF
import torch

target_gb = ${TARGET_GB}
device = torch.device("cuda:0")  # 映射后，这里写 0 即代表 GPU_ID

# float32 占用 4 字节
elements = target_gb * 1024**3 // 4

print(f"Allocating ~{target_gb} GB on GPU {${GPU_ID}} ...")
try:
    # 用 ones 保证显存真的被物理占用
    x = torch.ones(elements, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    print(f"Allocated {target_gb} GB on GPU {${GPU_ID}}. Press Ctrl+C to release.")
    while True:
        pass
except RuntimeError as e:
    print("Allocation failed:", e)
EOF
