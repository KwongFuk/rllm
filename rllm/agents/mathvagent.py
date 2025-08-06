import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class MathVAgent(BaseAgent):
    """
    A math agent that solves math problems (including multimodal: text + image) step-by-step.
    """

    def __init__(self, accumulate_thinking=True):
        self.instruction = "Let's think step by step, and put your final answer within \\boxed{}."
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state."""
        if not self.trajectory.steps:
            assert isinstance(observation, dict) and "question" in observation
            question = observation["question"]

            # 如果是多模态格式（list of dict）
            if isinstance(question, list):
                formatted_content = copy.deepcopy(question)
                # 给文字部分加推理指令
                for item in formatted_content:
                    if item.get("type") == "text":
                        item["text"] += f" {self.instruction}"
            else:
                # 纯文字问题
                formatted_content = [{"type": "text", "text": f"{question} {self.instruction}"}]
        else:
            # 答错时的修正提示
            formatted_content = [
                {
                    "type": "text",
                    "text": "Your previous answer may contain a mistake. Please review it carefully "
                            "and answer again. Put your final answer within \\boxed{}."
                }
            ]

        self.messages.append({"role": "user", "content": formatted_content})

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Update agent state based on model's response."""
        self.messages.append({"role": "assistant", "content": response})
        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions))
        self.trajectory.steps.append(new_step)
        return Action(action=response)

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, Any]]:
        """Return conversation history for model interaction."""
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant" and isinstance(msg["content"], str):
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Return current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
