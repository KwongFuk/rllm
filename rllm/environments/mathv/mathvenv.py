import warnings
from typing import Any

from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import RewardFunction, zero_reward


class MathVEnv(SingleTurnEnvironment):
    """
    数学图文推理环境
    基于 SingleTurnEnvironment 扩展，支持题干 + 图片 的单轮推理任务
    """

    def __init__(self, task: dict | None = None, reward_fn: RewardFunction | None = None, **kwargs):
        """
        Args:
            task: dict，至少包含：
                - question (str): 数学题文字
                - image (Any): 图片数据（路径、Base64字符串或图像对象）
                - answer (str|float|int): 正确答案
            reward_fn: 用于计算答案正确性的奖励函数
        """
        super().__init__(task=task, reward_fn=reward_fn, **kwargs)
        if reward_fn is None:
            warnings.warn("未提供奖励函数，使用零奖励", stacklevel=2)
        self.reward_fn = reward_fn or zero_reward

    def get_observation(self) -> dict:
        """
        返回 observation，包含题干和图片
        """
        obs = {}
        if self.task:
            obs["question"] = self.task.get("question", "")
            obs["options"] = self.task.get("options", "")
            obs["image"] = self.task.get("image", None)
        return obs

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        """
        根据任务和模型的答案计算奖励
        """
        reward_output = self.reward_fn(task_info=task, action=action)
        return reward_output.reward, {}

    @staticmethod
    def from_dict(env_args: dict) -> "MathVEnv":
        """
        从字典创建环境
        """
        reward_fn = env_args.pop("reward_fn", None)
        if "task" in env_args:
            task = env_args["task"]
        else:
            task = env_args
        return MathVEnv(task=task, reward_fn=reward_fn)
