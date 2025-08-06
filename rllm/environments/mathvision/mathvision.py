from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
from typing import Any


class MathVisionEnv(MultiTurnEnvironment):
    """
    环境：数学图文推理任务（支持多轮交互）。
    任务数据 task 示例:
    {
        "question": "已知下图三角形ABC...",
        "image_path": "/path/to/triangle.png",
        "ground_truth": "\\boxed{10}",
        "data_source": "MATHVISION"
    }
    """

    def __init__(
        self,
        task: dict | None = None,
        max_turns: int = 2,
        reward_bonus_coeff: float = 0.0,
        **kwargs
    ):
        super().__init__(task=task, max_turns=max_turns, **kwargs)
        self.reward_fn = None  # 外部传入的奖励计算函数
        self.prev_reward: float | None = None
        self.reward_bonus_coeff = reward_bonus_coeff

    def reset(self, task=None, seed=None):
        """重置环境并返回初始观测（支持图片+文字）。"""
        import random
        if seed is not None:
            random.seed(seed)

        if task is not None:
            self.task = task
        assert self.task is not None, "Task 必须在 reset 前设置"

        self.done = False
        self.current_turn = 0
        self.history = []
        self.prev_reward = None

        # 返回初始观测，支持多模态
        obs = {
            "question": self.task["question"],
            "image_path": self.task.get("image_path")  # 可选
        }
        return obs, {}

    def step(self, action: str):
        """
        执行一步推理：
        action: LLM 的回答（可能包含逐步推理）。
        """
        self.history.append(action)

        assert self.task is not None, "Task 未设置"
        raw_reward, next_obs = self.get_reward_and_next_obs(self.task, action)

        # 奖励 shaping
        if self.prev_reward is None:
            reward = raw_reward
        else:
            bonus = self.reward_bonus_coeff * (raw_reward - self.prev_reward)
            reward = raw_reward + bonus

        self.prev_reward = raw_reward
        self.current_turn += 1

        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    def get_reward_and_next_obs(self, task: dict, action: str) -> tuple[float, dict]:
        """
        调用奖励函数（比如比对答案、OCR 或数学推理评测）。
        """
        assert self.reward_fn is not None, "必须先设置 reward_fn"
        reward_response = self.reward_fn(
            data_source=task.get("data_source", ""),
            llm_solution=action,
            ground_truth=task["ground_truth"],
            image_path=task.get("image_path")
        )

        # 下一轮输入可以是反馈提示
        next_obs = {
            "feedback": reward_response.metadata.get("feedback", "请检查你的推理过程并再次作答"),
            "image_path": task.get("image_path")  # 图像可重复使用
        }
        return reward_response.reward, next_obs

    @staticmethod
    def from_dict(env_args: dict) -> "MathVisionEnv":
        return MathVisionEnv(
            task=env_args["task"],
            max_turns=env_args.get("max_turns", 2),
            reward_bonus_coeff=env_args.get("reward_bonus_coeff", 0.0)
        )
