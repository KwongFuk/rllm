import copy
import base64
from typing import Any, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class MathImageAgent(BaseAgent):
    """
    A math reasoning agent that can handle both textual and image-based problems.
    """

    def __init__(self, accumulate_thinking=True):
        self.instruction = (
            "Let's think step by step, and put your final answer within \\boxed{}."
        )
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 string for model input."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs
    ):
        """
        Process environment feedback and update internal state.
        observation can contain:
            {
                "question": str,
                "image_path": Optional[str]  # path to an image file
            }
        """
        if not self.trajectory.steps:
            assert isinstance(observation, dict) and "question" in observation
            question = observation["question"]

            if "image_path" in observation and observation["image_path"]:
                # Encode the image and add as multimodal input
                image_b64 = self._encode_image(observation["image_path"])
                formatted_observation = {
                    "type": "multimodal",
                    "content": [
                        {"type": "image", "data": image_b64},
                        {"type": "text", "text": f"{question} {self.instruction}"}
                    ]
                }
            else:
                # Pure text problem
                formatted_observation = f"{question} {self.instruction}"

        else:
            formatted_observation = (
                "Your previous answer may contain a mistake. "
                "Please review it carefully and answer again. "
                "Put your final answer within \\boxed{}."
            )

        self.messages.append({"role": "user", "content": formatted_observation})

    def update_from_model(self, response: str, **kwargs) -> Action:
        self.messages.append({"role": "assistant", "content": response})
        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions))
        self.trajectory.steps.append(new_step)
        return Action(action=response)

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, \
            "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
