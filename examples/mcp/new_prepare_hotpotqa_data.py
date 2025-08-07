import json
from datasets import Dataset
from rllm.data.dataset import DatasetRegistry


def prepare_hotpotqa_data(train_size=None, test_size=None):
    """
    Loading HotpotQA dataset from local JSON and registering it with the DatasetRegistry.
    Only loads essential fields: question, ground_truth, data_source

    Args:
        train_size: (Unused) Maximum number of training examples to load
        test_size: Maximum number of test examples to load

    Returns:
        tuple: (train_dataset, test_dataset)
    """

    def process_split(split_data, max_size):
        """Process a data split with optional size limit"""
        if max_size is not None:
            split_data = split_data.select(range(min(max_size, len(split_data))))
        processed = [
            {
                "question": example["question"],
                "ground_truth": example["answer"],
                "data_source": "hotpotqa"
            }
            for example in split_data
        ]
        print(f"Processed {len(processed)} examples")
        return processed

    print("Loading HotpotQA dataset from local file...")

    # 加载本地 JSON 文件
    with open("/home/yue/guangfu/data/hotpot/hotpot_dev_distractor_v1.json", "r") as f:
        raw_data = json.load(f)

    # 清洗字段，构造简洁结构，只保留 question 和 answer
    clean_data = [{"question": item["question"], "answer": item["answer"]} for item in raw_data]

    # 构造 HuggingFace Dataset 对象
    hotpot_dataset = {"validation": Dataset.from_list(clean_data)}

    # 处理 validation 集作为 test 集
    test_processed = process_split(hotpot_dataset["validation"], 10)

    train_dataset = None
    test_dataset = DatasetRegistry.register_dataset("hotpotqa", test_processed, "test")

    return None, test_dataset.get_data()


if __name__ == "__main__":
    _, test_dataset = prepare_hotpotqa_data(test_size=5)
    print("Test dataset:1", test_dataset[0])

    for i, item in enumerate(test_dataset):
        print(f"Example {i + 1}: {item}")

