import asyncio

from transformers import AutoProcessor
from rllm.agents.mathvagent import MathVAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.mathv.mathvenv import MathVEnv
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 2

    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    processor = AutoProcessor.from_pretrained(model_name)
    print(type(processor))
    
    agent_args = {
        "tools": [],
        "parser_name": "qwen",
        "system_prompt": (
            "You are a helpful math reasoning assistant. "
            "You will receive an image and a question about the image. "
            "Understand the visual content and answer the question concisely."
        )
    }

    env_args = {
        "tools": [],
        "reward_fn": math_reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=MathVAgent,
        agent_args=agent_args,
        env_class=MathVEnv,
        env_args=env_args,
        engine_name="openai",
        model_type="vl",
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "None"},
        processor=processor,
        sampling_params=sampling_params,
        max_response_length=16384,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("MathVision", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_mathvision_data import prepare_math_data

        _, test_dataset = prepare_math_data()

    tasks = test_dataset.repeat(n=2)  # repeat to evaluate pass@k

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)