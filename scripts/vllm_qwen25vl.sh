#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
# 单机多卡建议先别关 P2P
unset NCCL_P2P_DISABLE
unset NCCL_IB_DISABLE

PORT=30000
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

LIMIT_MM='{"image": 3, "video": 0}'
# 这个参数 Qwen2VL 会忽略，去掉就不会有 warning
MM_KWARGS='{}'

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port ${PORT} \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt "${LIMIT_MM}" \
  --mm-processor-kwargs "${MM_KWARGS}" \
  --swap-space 16 \
  --disable-custom-all-reduce \
  --enforce-eager


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

