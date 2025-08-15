import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import zipfile

# 必须放在最前面（如果需要代理）
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# os.environ["HTTP_PROXY"] = "socks5h://localhost:7897"
# os.environ["HTTPS_PROXY"] = "socks5h://localhost:7897"

cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
print("默认缓存路径:", cache_dir)

# 加载数据集
ds = load_dataset("MathLLMs/MathVision", token=True, cache_dir=cache_dir)
print(ds)

# 打印每个 split 对应的缓存文件信息
for split, dataset in ds.items():
    print(f"\n=== Split: {split} ===")
    for cache in dataset._info.download_checksums:
        print("缓存文件:", cache)


