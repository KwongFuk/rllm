"""
prepare_mathvision_data.py

This script prepares and processes two datasets for math-related vision-language tasks:

1. MathV360K (source: Zhiqiang007/MathV360K) - A large-scale math visual question answering dataset.
   - Loaded in streaming mode, the first 1000 samples are taken for training.
   - Each sample is processed by `preprocess_mathv360k_fn` to match a unified format
     with fields like id, question, options, image, decoded_image, answer, solution, level, subject.

2. MathVision (source: MathLLMs/MathVision, split: testmini) - A smaller dataset used for testing.
   - Loaded in full without streaming.

The processed datasets are registered into the RLLM DatasetRegistry:
    - "Zhiqiang007/MathV360K" -> train split
    - "MathLLMs/MathVision" -> test split

Usage:
    python prepare_mathvision_data.py

Dependencies:
    - datasets (Hugging Face)
    - rllm.data.dataset.DatasetRegistry
    - re, itertools
"""

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry
import re
import os

def preprocess_mathv360k_fn(example, idx):
    """
    Preprocess a single MathV360K example into the unified target format.
    Ensures all fields have consistent types and extracts multiple-choice options (A/B/C/D...).
    """
    # Extract human question
    human_text = ""
    for conv in example.get("conversations", []):
        if conv.get("from") == "human":
            human_text = conv.get("value", "")
            break

    # Extract question part
    if "Question:" in human_text:
        question_part = human_text.split("Question:")[-1].strip()
    else:
        question_part = human_text.strip()

    # Extract options
    options = []
    if "Choices:" in question_part:
        # Split question and choices part
        q_text, choices_text = question_part.split("Choices:", 1)
        question = q_text.strip()

        # Regex match for (A) Option text
        options = re.findall(r"\([A-Z]\)\s*([^\n]+)", choices_text)
        options = [opt.strip() for opt in options]
    else:
        question = question_part

    # Extract GPT answer
    answer_text = ""
    for conv in example.get("conversations", []):
        if conv.get("from") == "gpt":
            ans_text = conv.get("value", "").strip()
            if ans_text.lower().startswith("the answer is"):
                ans_text = ans_text[len("the answer is"):].strip()
            answer_text = ans_text
            break

    # Return standardized record
    return {
        "id": str(idx),
        "question": question,
        "options": options,
        "image": example.get("image", ""),
        "decoded_image": "",
        "groundtruth": str(answer_text),
        "solution": None,
        "level": None,
        "subject": None,
        "data_source":"mathv360k"
    }

def preprocess_mathvision_fn(sample):
    """
    将样本转换为统一格式：
    - 将 'answer' 重命名为 'groundtruth'
    - 添加 'data_source' 字段
    - 基于原 'image' 字段生成 'image_url_local'
    - 保留其他字段
    """
    new_sample = sample.copy()

    # 修改字段名
    new_sample["groundtruth"] = new_sample.pop("answer")

    # 添加新字段
    new_sample["data_source"] = "MathVision"

    # 基于原 image 字段拼接本地路径
    base_path = "/home/smm/.cache/huggingface/datasets/MathLLMs/MathVision-images/"
    if "image" in new_sample:
        new_sample["image_url_local"] = os.path.join(base_path, new_sample["image"])

    return new_sample



def prepare_math_data():
    # Load test dataset (MathVision)
    test_dataset = load_dataset("MathLLMs/MathVision", split="testmini")
    test_dataset = test_dataset.map(preprocess_mathvision_fn)

    test_dataset = test_dataset.select(range(5))
    
    # Load train dataset (MathV360K) in streaming mode
    # dataset_stream = load_dataset("Zhiqiang007/MathV360K", split="train", streaming=True)

    train_dataset = None
    # Take first 1000 samples
    # train_dataset = Dataset.from_list(list(islice(dataset_stream, 1000)))
    # train_dataset = train_dataset.map(preprocess_mathv360k_fn, with_indices=True)

    # # Register datasets
    # train_dataset = DatasetRegistry.register_dataset("MathV360K", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("MathVision", test_dataset, "test")
    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = prepare_math_data()

    # # Print first train sample
    # print("=== Train dataset first sample ===")
    # for key, value in train_dataset[0].items():
    #     print(f"{key}: {value}")

    # Print first test sample
    print(f"Test dataset size: {len(test_dataset)} samples")
    print("\n=== Test dataset first sample ===")
    for key, value in test_dataset[0].items():
        print(f"{key}: {value}")
