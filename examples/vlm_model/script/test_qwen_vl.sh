#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

API_URL="http://127.0.0.1:30000/v1/chat/completions"
MODEL_PATH="/home/smm/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"
IMAGE1_PATH="/home/smm/.cache/huggingface/datasets/demo.jpeg"
IMAGE2_PATH="/home/smm/.cache/huggingface/datasets/MathLLMs/MathVision-images/images/4.jpg"


echo "[INFO] 测试文本对话..."
curl -s "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_PATH\",
    \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}]
  }"

echo -e "\n[INFO] 测试多模态对话(本地路径)..."
curl -s "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_PATH\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"file://$IMAGE1_PATH\"}},
          {\"type\": \"text\", \"text\": \"请详细描述这张图片的内容\"}
        ]
      }
    ],
    \"max_tokens\": 128
  }"

echo -e "\n[INFO] 测试多模态对话(本地路径2)..."
curl -s "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_PATH\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"file://$IMAGE2_PATH\"}},
          {\"type\": \"text\", \"text\": \"请详细描述这张图片的内容\"}
        ]
      }
    ],
    \"max_tokens\": 128
  }"