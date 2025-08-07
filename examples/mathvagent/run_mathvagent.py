import asyncio

from transformers import AutoTokenizer

from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 1

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    agent_args = {"tools": ["python"], "parser_name": "qwen", "system_prompt": "You are a math assistant that can write python to solve math problems."}
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=ToolAgent,
        agent_args=agent_args,
        env_class=ToolEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "None"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=16384,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("mathvision", "test")
    

    tasks = test_dataset.repeat(n=1)  # repeat to evaluate pass@k

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
