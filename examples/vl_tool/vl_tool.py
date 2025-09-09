import asyncio
import os
import logging
from transformers import AutoProcessor

import torch

from rllm.agents import ICOTAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments import ICOTEnvironment
from rllm.rewards.reward_fn import multi_modal_reward_fn
from rllm.utils import compute_pass_at_k

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # 配置参数
    n_parallel_agents = 4
    model_name = "Qwen/Qwen-VL-Chat"  # 使用多模态模型
    max_steps = 10
    signal_token = "\n"  # 特殊符号
    
    # 加载模型和分词器
    logger.info(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # model_name = "Qwen/Qwen3-4B"
    model_name = "/home/smm/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    
    # 配置Agent参数
    agent_args = {
        "model": model,
        "tokenizer": tokenizer,
        "parser_name": "qwen",
        "system_prompt": "你是一个多模态助手，可以分析图像内容并生成相关描述。",
        "signal_token": signal_token,
        "max_new_tokens": 256,
        "temperature": 0.7
    }
    
    # 配置环境参数
    env_args = {
        "tools": ["image_embed_selection", "python"],
        "reward_fn": multi_modal_reward_fn,
        "max_steps": max_steps,
        "model": model,
        "tokenizer": tokenizer,
        "signal_token": signal_token,
        "embed_dim": model.config.hidden_size
    }
    
    # 配置采样参数
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "model": model_name
    }
    
    # 初始化执行引擎
    engine = AgentExecutionEngine(
        agent_class=ICOTAgent,
        agent_args=agent_args,
        env_class=ICOTEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None"
        },
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=16384,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )
    
    # 加载或创建测试数据集
    try:
        test_dataset = DatasetRegistry.load_dataset("multimodal_test", "test")
    except:
        logger.info("Dataset not found, creating sample dataset...")
        test_dataset = [
            {
                "question": "描述这张图片中的内容。",
                "image_url_local": "test_image.jpg",
                "answer": "这是一张包含...的图片"
            }
        ]
    
    # 执行任务
    logger.info(f"Starting evaluation with {len(test_dataset)} tasks")
    results = asyncio.run(engine.execute_tasks(test_dataset))
    
    # 计算结果
    pass_at_k = compute_pass_at_k(results)
    logger.info(f"Evaluation results: {pass_at_k}")
    