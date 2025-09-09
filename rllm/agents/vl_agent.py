import copy
import json
import logging
import uuid
from typing import Any, Tuple, Dict, List

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import TOOL_SYSTEM_PROMPT
from rllm.parser import get_tool_parser
from rllm.parser.tool_parser.tool_parser_base import ToolParser
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool

logger = logging.getLogger(__name__)

class ImageEmbedSelectionTool(Tool):
    """图像嵌入选择工具，用于从原始图像嵌入中选择与文本语义匹配的嵌入"""
    name = "image_embed_selection"
    description = (
        "选择与文本语义匹配的图像嵌入片段，基于文本隐藏状态与图像嵌入的相似度"
    )

    @property
    def json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "text_hidden_state": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "文本最后一个token的隐藏状态"
                    },
                    "sim_threshold": {
                        "type": "number",
                        "default": 0.2,
                        "description": "相似度阈值"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 32,
                        "description": "最大选择数量"
                    }
                },
                "required": ["text_hidden_state"]
            }
        }

    def run(self, text_hidden_state: list[float], sim_threshold: float = 0.2, top_k: int = 32) -> dict:
        try:
            return {
                "status": "success",
                "text_hidden_state": text_hidden_state,
                "sim_threshold": sim_threshold,
                "top_k": top_k
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}


class ICOTAgent(BaseAgent):
    """ICOT Agent，负责管理对话状态和决策"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = TOOL_SYSTEM_PROMPT,
        parser_name: str = "qwen",
        tool_map: Dict[str, type[Tool]] | None = None,
        signal_token: str = "\n",
        max_new_tokens: int = 256,
        temperature: float = 1.0
    ):
        # 初始化工具
        default_tools = {ImageEmbedSelectionTool.name: ImageEmbedSelectionTool}
        if tool_map:
            default_tools.update(tool_map)
        
        self.tools = MultiTool(tool_map=default_tools)
        
        # 初始化解析器
        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()
        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(self.tools.json, indent=2))
        
        # 模型和分词器
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # 超参数
        self.signal_token = signal_token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # 状态管理
        self._trajectory = Trajectory()
        self.messages: List[dict[str, Any]] = []
        self.current_observation = None
        self.image_embeds: torch.Tensor | None = None  # 存储图像嵌入
        self.generated_text = ""  # 累积生成的文本
        self.current_step = 0  # 当前生成步骤
        self.waiting_for_hidden_state = False  # 是否在等待hidden state
        self.inputs_embeds: torch.Tensor | None = None  # 输入嵌入
        self.inputs: Dict[str, torch.Tensor] | None = None  # 输入ids
        
        # 重置状态
        self.reset()

    def reset(self):
        """重置Agent状态"""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt + self.tools_prompt}]
        self.current_observation = None
        self.image_embeds = None
        self.generated_text = ""
        self.current_step = 0
        self.waiting_for_hidden_state = False
        self.inputs_embeds = None
        self.inputs = None

    def _format_observation_as_messages(self, obs: Any) -> List[dict]:
        """格式化观测结果为消息"""
        messages = []
        
        if isinstance(obs, dict):
            # 处理工具输出
            if "tool_outputs" in obs:
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append({
                        "role": "tool",
                        "content": tool_output_str,
                        "tool_call_id": tool_call_id
                    })
            
            # 处理环境返回的生成结果
            if "generated_text" in obs:
                self.generated_text += obs["generated_text"]
                messages.append({
                    "role": "assistant",
                    "content": obs["generated_text"]
                })
            
            # 处理图像嵌入
            if "image_embeds" in obs:
                self.image_embeds = obs["image_embeds"]
            
            # 处理输入文本
            if "question" in obs and not self.inputs_embeds:
                text = obs["question"]
                messages.append({"role": "user", "content": text})
                
                # 准备文本嵌入
                self.inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True
                ).to(self.device)
                self.inputs_embeds = self.model.get_input_embeddings()(self.inputs["input_ids"])
        
        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """从环境更新状态"""
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        self.current_observation = observation
        
        # 检查是否收到hidden state
        if "hidden_state" in observation and "generated_text" in observation:
            self.waiting_for_hidden_state = False
            # 检查生成的文本中是否包含特殊符号
            if self.signal_token in observation["generated_text"]:
                logger.info(f"Detected signal token '{self.signal_token}', preparing to select image embeds")

    def update_from_model(self, response: dict,** kwargs) -> Action:
        """根据模型响应更新状态并生成动作"""
        # 如果还在生成过程中且未达到最大步骤
        if self.current_step < self.max_new_tokens and not response.get("done", False):
            self.current_step += 1
            
            # 如果等待hidden state，调用图像嵌入选择工具
            if self.waiting_for_hidden_state and self.image_embeds is not None:
                tool_call_id = str(uuid.uuid4())
                tool_action = [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": ImageEmbedSelectionTool.name,
                        "arguments": json.dumps({
                            "text_hidden_state": response["hidden_state"],
                            "sim_threshold": 0.2,
                            "top_k": 32
                        })
                    }
                }]
                
                self.messages.append({
                    "role": "assistant",
                    "content": f"调用工具选择图像嵌入...",
                    "tool_calls": tool_action
                })
                
                self._trajectory.steps.append(Step(
                    chat_completions=copy.deepcopy(self.messages),
                    action=tool_action,
                    model_response="选择图像嵌入",
                    observation=self.current_observation
                ))
                
                return Action(action=tool_action)
            
            # 继续生成
            continue_action = [{
                "type": "continue_generation",
                "parameters": {
                    "temperature": self.temperature
                }
            }]
            
            self.messages.append({
                "role": "assistant",
                "content": response.get("generated_text", "")
            })
            
            self._trajectory.steps.append(Step(
                chat_completions=copy.deepcopy(self.messages),
                action=continue_action,
                model_response=response,
                observation=self.current_observation
            ))
            
            return Action(action=continue_action)
        
        # 完成生成
        finish_action = [{
            "type": "function",
            "function": {
                "name": "finish",
                "arguments": {
                    "response": self.generated_text
                }
            }
        }]
        
        self.messages.append({
            "role": "assistant",
            "content": self.generated_text
        })
        
        self._trajectory.steps.append(Step(
            chat_completions=copy.deepcopy(self.messages),
            action=finish_action,
            model_response=self.generated_text,
            observation=self.current_observation
        ))
        
        return Action(action=finish_action)

    @property
    def chat_completions(self) -> List[dict[str, str]]:
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
    