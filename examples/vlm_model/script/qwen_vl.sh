#!/usr/bin/env bash
set -euo pipefail

# ==== 配置部分 ====
export CUDA_VISIBLE_DEVICES=2
unset NCCL_P2P_DISABLE
unset NCCL_IB_DISABLE

PORT=30000
MODEL="/home/smm/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"

LIMIT_MM='{"image": 3, "video": 0}'
MM_KWARGS='{}'

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/qwen7b_api_$(date +%F_%H-%M-%S).log"

# ==== 清理 GPU 占用 ====
echo "[INFO] 检查 GPU ${CUDA_VISIBLE_DEVICES} 占用..."
PIDS=$(nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --query-compute-apps=pid --format=csv,noheader | grep -v '^$' || true)
if [ -n "$PIDS" ]; then
  echo "[INFO] 杀掉进程: $PIDS"
  kill -9 $PIDS
  sleep 2
else
  echo "[INFO] GPU ${CUDA_VISIBLE_DEVICES} 没有占用进程"
fi

# ==== 启动服务 ====
echo "[INFO] 启动 vLLM API Server"
echo "[INFO] 模型路径: ${MODEL}"
echo "[INFO] 端口: ${PORT}"
echo "[INFO] GPU: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] 日志文件: ${LOG_FILE}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port ${PORT} \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt "${LIMIT_MM}" \
  --mm-processor-kwargs "${MM_KWARGS}" \
  --swap-space 16 \
  --disable-custom-all-reduce \
  --enforce-eager \
  --allowed-local-media-path /home/smm/ggf/rllm/examples/vlm_model \
  2>&1 | tee -a "${LOG_FILE}"




# curl http://127.0.0.1:30000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen/Qwen2.5-VL-7B-Instruct",
#     "messages": [{"role": "user", "content": "你好"}]
#   }'


# curl http://127.0.0.1:30000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen/Qwen2.5-VL-7B-Instruct",
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#           { "type": "image_url", "image_url": { "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" } },
#           { "type": "text", "text": "请详细描述这张图片的内容" }
#         ]
#       }
#     ],
#     "max_tokens": 128
#   }'