import base64
import asyncio
import os
import logging
from openai import AsyncOpenAI, RateLimitError
from rllm.data.dataset import DatasetRegistry  # 假设你自己有这个模块

logger = logging.getLogger(__name__)

def convert_image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """
    将图像字节数据转换为 OpenAI 支持的 base64 data URI。
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{base64_image}"


async def ask_openai_gpt4o_with_image_async(image_url: str, prompt: str = "请描述这张图像的内容。", application_id = 0, retries: int = 2) -> str:
    client = AsyncOpenAI()

    while retries > 0:
        try:
            response = await client.responses.create(
                model="gpt-4o-mini",  # 或 "gpt-4o"
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",  # ✅ 正确的文本类型
                                "text": prompt,
                            },
                            {
                                "type": "input_image",  # ✅ 正确的图像类型
                                "image_url": image_url  
                            }
                        ]
                    }
                ]
            )
            return response.output_text
        except RateLimitError:
            retries -= 1
            if retries == 0:
                return "调用失败：达到速率限制，重试次数耗尽。"
            logger.warning("速率限制，5秒后重试...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error("调用 OpenAI 出错: %s", e)
            return f"调用 OpenAI 失败: {e}"



async def main():
    print("🔑 当前 API Key:", os.getenv("OPENAI_API_KEY"))
    
    # ✅ 加载测试数据
    test_dataset = DatasetRegistry.load_dataset("MathVision", "test")
    sample = test_dataset[0]
    image_bytes = sample["decoded_image"]["bytes"]
    question = sample["question"]

    # ✅ 转换图像
    image_url = convert_image_bytes_to_data_url(image_bytes)
    application_id = 0
    # ✅ 调用 GPT 模型识别图像
    prompt = question
    print("发送图像给 GPT-4o 分析中...")
    result = await ask_openai_gpt4o_with_image_async(image_url, prompt, application_id)
    print("🧠 GPT-4o 响应：", result)


if __name__ == "__main__":
    asyncio.run(main())
