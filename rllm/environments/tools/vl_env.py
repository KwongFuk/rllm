import json
import queue
import threading
import warnings
from typing import Any, Dict, List, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class ICOTEnvironment(BaseEnv):
    """ICOT Environment，负责模型推理和嵌入处理"""
    
    def __init__(
        self, 
        task: dict | None = None, 
        tools: list[str] | None = None, 
        tool_map: dict[str, type[Tool]] | None = None, 
        reward_fn: RewardFunction | None = None, 
        max_steps: int = 10,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer = None,
        signal_token: str = "\n",
        embed_dim: int = 768
    ):
        if tool_map is not None and tools is not None:
            raise ValueError("不能同时指定'tools'和'tool_map'参数")
        
        self.step_count = 0
        self.max_steps = max_steps
        self.signal_token = signal_token
        self.embed_dim = embed_dim
        
        # 初始化工具
        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])
        
        # 模型和分词器
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device if model else torch.device("cpu")
        
        # 任务和奖励函数
        self.task = task
        self.reward_fn = reward_fn if reward_fn is not None else zero_reward
        
        # 状态管理
        self.current_embeds: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.position_ids: torch.Tensor | None = None
        self.generated_ids: List[int] = []
        self.image_embeds: torch.Tensor | None = None  # 原始图像嵌入
        self.selected_embeds: torch.Tensor | None = None  # 选择的图像嵌入
        self.used_patch_ids = set()  # 已使用的图像patch ID
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def reset(self) -> Tuple[dict, dict]:
        """重置环境状态"""
        self.step_count = 0
        self.current_embeds = None
        self.attention_mask = None
        self.position_ids = None
        self.generated_ids = []
        self.selected_embeds = None
        self.used_patch_ids = set()
        
        # 处理任务中的图像
        if self.task and "image_url_local" in self.task:
            self.image_embeds = self._generate_image_embeds(self.task["image_url_local"])
            obs = {**self.task, "image_embeds": self.image_embeds}
        else:
            obs = self.task if self.task else {}
            
        return obs, {}

    def _generate_image_embeds(self, image_path: str) -> torch.Tensor:
        """生成图像嵌入"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # 生成图像嵌入（假设模型有此方法）
            if hasattr(self.model, "get_image_embeddings"):
                image_embeds = self.model.get_image_embeddings(image_tensor)
            else:
                #  fallback方法，使用特征提取器
                with torch.no_grad():
                    outputs = self.model.visual(image_tensor)
                    image_embeds = outputs.last_hidden_state
                
            # 确保输出形状为 [N, D]
            if image_embeds.dim() == 3 and image_embeds.size(0) == 1:
                image_embeds = image_embeds.squeeze(0)
                
            logger.info(f"Generated image embeddings with shape: {image_embeds.shape}")
            return image_embeds
        except Exception as e:
            logger.error(f"Failed to generate image embeddings: {str(e)}")
            raise

    def step(self, action: List[dict] | str | dict) -> Tuple[dict, float, bool, dict]:
        """执行一步环境交互"""
        if action is None:
            action = []
            
        if isinstance(action, dict):
            action = [action]
            
        self.step_count += 1
        reward = 0.0
        done = self.step_count >= self.max_steps
        
        # 检查是否是结束动作
        if isinstance(action, list) and action:
            for tool_call in action:
                if tool_call.get("function", {}).get("name") == "finish":
                    done = True
                    break
        
        # 如果完成，计算奖励
        if done:
            if isinstance(action, str):
                response = action
            elif isinstance(action, list):
                finish_action = next(
                    (tc for tc in action if tc.get("function", {}).get("name") == "finish"),
                    None
                )
                if finish_action:
                    arguments = finish_action.get("function", {}).get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    response = arguments.get("response", "")
                else:
                    response = str(action)
            else:
                response = str(action)
                
            task_info = self.task if self.task is not None else {}
            reward_output = self.reward_fn(task_info=task_info, action=response)
            return {}, reward_output.reward, done, {
                "response": response, 
                "metadata": reward_output.metadata
            }
        
        # 处理工具调用
        if isinstance(action, list) and action and action[0].get("type") == "function":
            tool_outputs = self._execute_tool_calls(action)
            next_obs = {"tool_outputs": tool_outputs}
            
            # 如果是图像嵌入选择工具，处理结果并拼接嵌入
            for tool_id, output_str in tool_outputs.items():
                output = json.loads(output_str)
                if output.get("status") == "success" and "text_hidden_state" in output:
                    if self.image_embeds is not None:
                        # 执行图像嵌入选择
                        selected_embeds, patch_ids = self._select_image_embeds(
                            text_hidden_state=torch.tensor(output["text_hidden_state"], device=self.device),
                            sim_threshold=output.get("sim_threshold", 0.2),
                            top_k=output.get("top_k", 32)
                        )
                        self.selected_embeds = selected_embeds
                        self.used_patch_ids.update(patch_ids)
                        
                        # 拼接嵌入
                        if self.current_embeds is not None and self.selected_embeds is not None:
                            # 获取信号 token 嵌入
                            signal_token_id = self.tokenizer.encode(self.signal_token, add_special_tokens=False)[0]
                            signal_embed = self.model.get_input_embeddings()(
                                torch.tensor([signal_token_id], device=self.device)
                            ).unsqueeze(0)
                            
                            # 拼接嵌入：信号 token + 选择的图像嵌入 + 信号 token
                            new_embeds = torch.cat([
                                signal_embed,
                                self.selected_embeds.unsqueeze(0),
                                signal_embed
                            ], dim=1)
                            
                            # 更新当前嵌入
                            self.current_embeds = torch.cat([self.current_embeds, new_embeds], dim=1)
                            # 更新注意力掩码
                            self.attention_mask = torch.cat([
                                self.attention_mask,
                                torch.ones((1, new_embeds.size(1)), dtype=torch.long, device=self.device)
                            ], dim=1)
                            # 更新位置ID
                            self.position_ids = torch.arange(
                                self.current_embeds.shape[1], device=self.device
                            ).unsqueeze(0)
            
            return next_obs, reward, done, {"response": action, "metadata": {}}
        
        # 处理继续生成动作
        if isinstance(action, list) and action and action[0].get("type") == "continue_generation":
            return self._continue_generation(action[0]["parameters"])
        
        # 默认返回
        return {}, reward, done, {"response": action, "metadata": {}}

    def _select_image_embeds(self, text_hidden_state: torch.Tensor, sim_threshold: float, top_k: int) -> Tuple[torch.Tensor, List[int]]:
        """选择与文本隐藏状态相似的图像嵌入"""
        def l2norm(x, eps=1e-8):
            return x / (x.norm(dim=-1, keepdim=True) + eps)
        
        # 归一化
        norm_text = l2norm(text_hidden_state.unsqueeze(0))  # [1, D]
        norm_image = l2norm(self.image_embeds)  # [N, D]
        
        # 计算相似度
        sim_scores = torch.matmul(norm_text, norm_image.transpose(0, 1)).squeeze(0)  # [N]
        
        # 排除已使用的patch
        if self.used_patch_ids:
            used_ids = torch.tensor(list(self.used_patch_ids), device=self.device, dtype=torch.long)
            sim_scores[used_ids] = float("-inf")
        
        # 选择超过阈值的嵌入
        patch_ids = (sim_scores >= sim_threshold).nonzero(as_tuple=False).squeeze(1)
        
        # 如果没有足够的，使用top-k
        if patch_ids.numel() == 0:
            k = min(5, sim_scores.numel())
            _, patch_ids = sim_scores.topk(k=k, largest=True, sorted=False)
        
        # 限制最大数量
        if patch_ids.numel() > top_k:
            _, top_idx = sim_scores[patch_ids].topk(k=top_k, largest=True, sorted=False)
            patch_ids = patch_ids[top_idx]
        
        return self.image_embeds[patch_ids], patch_ids.tolist()

    def _continue_generation(self, parameters: Dict[str, Any]) -> Tuple[dict, float, bool, dict]:
        """继续生成文本，遇到特殊符号时返回hidden state"""
        if self.current_embeds is None and self.task and "question" in self.task:
            # 初始化生成
            text = self.task["question"]
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            self.current_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
            self.attention_mask = torch.ones_like(inputs["input_ids"], device=self.device)
            self.position_ids = torch.arange(self.current_embeds.shape[1], device=self.device).unsqueeze(0)
            self.generated_ids = inputs["input_ids"].squeeze().tolist()
        
        # 执行一步生成
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=self.current_embeds,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True
            )
        
        # 获取logits和hidden state
        logits = outputs.logits[:, -1, :]
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, D]
        
        # 采样下一个token
        temperature = parameters.get("temperature", 1.0)
        if temperature > 0:
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_token = torch.argmax(logits, dim=-1)
        
        token_id = int(next_token.item())
        self.generated_ids.append(token_id)
        token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        
        # 获取下一个token的嵌入
        next_token_embed = self.model.get_input_embeddings()(next_token).unsqueeze(1)
        
        # 更新嵌入、注意力掩码和位置ID
        self.current_embeds = torch.cat([self.current_embeds, next_token_embed], dim=1)
        self.attention_mask = torch.cat([
            self.attention_mask, 
            torch.ones((1, 1), dtype=torch.long, device=self.device)
        ], dim=1)
        self.position_ids = torch.arange(
            self.current_embeds.shape[1], device=self.device
        ).unsqueeze(0)
        
        # 检查是否遇到特殊符号
        contains_signal = self.signal_token in token_str
        done = token_id in [
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", -1)
        ]
        
        # 准备返回结果
        result = {
            "generated_text": token_str,
            "done": done
        }
        
        # 如果遇到特殊符号，返回hidden state
        if contains_signal:
            result["hidden_state"] = last_hidden.squeeze(0).tolist()
        
        return result, 0.0, done, {"metadata": {"token_id": token_id, "contains_signal": contains_signal}}

    def _execute_tool_calls(self, tool_calls: List[dict[Any, Any]]) -> Dict[str, str]:
        """执行工具调用"""
        tool_outputs: Dict[str, str] = {}
        output_queue: queue.Queue[Tuple[str, str]] = queue.Queue()
        threads = []
        
        def execute_tool(tool_call):
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_output = self.tools(tool_name=tool_name, **tool_args)
            tool_output_str = tool_output.to_string()
            output_queue.put((tool_call["id"], tool_output_str))
        
        # 为每个工具调用创建线程
        for tool_call in tool_calls:
            thread = threading.Thread(target=execute_tool, args=(tool_call,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        while not output_queue.empty():
            tool_call_id, output_str = output_queue.get()
            tool_outputs[tool_call_id] = output_str
        
        return tool_outputs

    @staticmethod
    def from_dict(env_args: dict) -> "ICOTEnvironment":
        tools = env_args.pop("tools", None)
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        model = env_args.pop("model", None)
        tokenizer = env_args.pop("tokenizer", None)
        signal_token = env_args.pop("signal_token", "\n")
        embed_dim = env_args.pop("embed_dim", 768)
        
        return ICOTEnvironment(
            task=env_args, 
            tools=tools, 
            tool_map=tool_map, 
            max_steps=max_steps, 
            reward_fn=reward_fn,
            model=model,
            tokenizer=tokenizer,
            signal_token=signal_token,
            embed_dim=embed_dim
        )
    