import asyncio
import os
import sys

from prepare_hotpotqa_data import prepare_hotpotqa_data
from transformers import AutoTokenizer

from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from rllm.agents.tool_agent import MCPToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.mcp_env import MCPConnectionManager, MCPEnvironment
from rllm.rewards.reward_fn import search_reward_fn
from rllm.utils import save_trajectories


async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tool_mcp.py <tavily_api_key>")
        print("This will run HotpotQA evaluation using Tavily MCP server")
        sys.exit(1)

    tavily_api_key = sys.argv[1]
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    n_parallel_agents = 1
    model_name = "gpt-4o-mini"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mcp_server_command = "npx"
    mcp_server_args = ["-y", "tavily-mcp@0.2.4"]
    mcp_server_env = {"TAVILY_API_KEY": tavily_api_key}

    temp_manager = MCPConnectionManager(mcp_server_command, mcp_server_args, mcp_server_env)
    temp_manager.start()
    try:
        mcp_tool_map = temp_manager.tool_map
        print(f"Available tools: {list(mcp_tool_map.keys())}")
    finally:
        temp_manager.stop()

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=MCPToolAgent,
        env_class=MCPEnvironment,
        agent_args={"parser_name": "qwen", "system_prompt": SEARCH_SYSTEM_PROMPT, "tool_map": mcp_tool_map},
        env_args={
            "mcp_server_command": mcp_server_command,
            "mcp_server_args": mcp_server_args,
            "mcp_server_env": mcp_server_env,
            "reward_fn": search_reward_fn,
        },
        engine_name="openai_v",
        rollout_engine_args={"base_url": "https://api.openai.com/v1", "api_key": "None"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=16384,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("MathVision", "test")

    tasks = test_dataset.get_data()[:10]
    print(f"Running evaluation on {len(tasks)} tasks...")

    try:
        results = await engine.execute_tasks(tasks)
        save_trajectories(results, save_dir="./trajectories/v_mcp_tavily", filename="trajectories.pt")
    finally:
        MCPEnvironment.cleanup_global_resources()


if __name__ == "__main__":
    asyncio.run(main())
