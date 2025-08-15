import asyncio

from transformers import AutoProcessor

from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k_v

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 2

    # model_name = "Qwen/Qwen3-4B"
    model_name = "/home/smm/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    # agent_args = {
    #         "tools": ["python"],
    #         "parser_name": "qwen",
    #         "system_prompt": """You are a math assistant that can write Python to solve math problems.
    #     You will often be given one or more images along with the problem statement.
    #     Carefully examine the image(s) to extract all relevant visual information 
    #     (such as text, numbers, objects, patterns, spatial relations) before answering.
    #     Describe what you see from the image in your own words, then use that information 
    #     together with the text of the problem to solve it step-by-step.
    #     If necessary, write Python code to compute the answer.
    #     """
    #     }


    agent_args = {
    "tools": [],  # 不传工具
    "parser_name": "qwen",
    "system_prompt": """You are a vision reasoning assistant.
You will often be given one or more images along with the problem statement.
Carefully examine the image(s) to extract all relevant visual information 
(such as text, numbers, objects, patterns, spatial relations) before answering.

Follow this procedure for every problem:
1. Describe in words what you see in the image.
2. Identify key numbers, symbols, or relationships from the image and text.
3. Reason step-by-step to solve the problem without executing any code.
4. State the final answer on a separate line in the format: ANSWER: <value>

Do not write or execute Python code.
"""
}


    env_args = {
        "tools": [],
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
        max_response_length=4096,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("MathVision", "test")
    # if test_dataset is None:
    #     print("Dataset not found, preparing dataset...")
    #     from prepare_math_data import prepare_math_data

    #     _, test_dataset = prepare_math_data()

    tasks = test_dataset.get_data()[:5]
    # tasks = test_dataset.repeat(n=1)  # repeat to evaluate pass@k

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k_v(results)
