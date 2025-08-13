#!/usr/bin/env bash

GPU_ID=2          # 要占的 GPU ID
RESERVE_MB=500    # 给系统预留的显存（MB）
SLEEP_INTERVAL=2  # 检测间隔秒数

OCCUPY_SCRIPT=$(cat <<EOF
import torch, subprocess, time, os, sys

GPU_ID = ${GPU_ID}
RESERVE_MB = ${RESERVE_MB}

def get_free_mem_mb(gpu_id):
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    )
    free_list = [int(x) for x in result.decode().strip().split("\\n")]
    return free_list[gpu_id]

while True:
    free_mb = get_free_mem_mb(GPU_ID)
    print(f"[GPU {GPU_ID}] 当前剩余显存 {free_mb} MB", flush=True)

    target_mb = max(free_mb - RESERVE_MB, 0)
    if target_mb > 0:
        try:
            print(f"[GPU {GPU_ID}] 尝试分配 ~{target_mb} MB", flush=True)
            elements = target_mb * 1024 * 1024 // 4  # float32: 4字节
            x = torch.ones(elements, dtype=torch.float32, device=f"cuda:{GPU_ID}")
            torch.cuda.synchronize()
            print(f"[GPU {GPU_ID}] 成功分配 {target_mb} MB，开始保持...", flush=True)
            while True:
                time.sleep(1)
        except RuntimeError as e:
            print(f"[GPU {GPU_ID}] 分配失败: {e}", flush=True)
            time.sleep(2)
    else:
        time.sleep(${SLEEP_INTERVAL})
EOF
)

while true; do
    if ! pgrep -f "gpu_guard_dynamic_alloc_${GPU_ID}" >/dev/null; then
        echo "[$(date)] GPU ${GPU_ID} 有空闲，启动占用进程..."
        python -u -c "$OCCUPY_SCRIPT" gpu_guard_dynamic_alloc_${GPU_ID} &
    fi
    sleep $SLEEP_INTERVAL
done
