import requests, zipfile, io, os

# 构造下载 URL
url = "https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/images.zip"

# 下载 zip 文件
print("正在下载 images.zip …")
response = requests.get(url)
response.raise_for_status()  # 若请求失败会抛出异常

# 创建解压目录
output_dir = os.path.expanduser("~/.cache/huggingface/datasets/MathLLMs/MathVision-images")
os.makedirs(output_dir, exist_ok=True)

# 解压 zip 内容
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(output_dir)

print("✅ 下载并解压完成，路径为：", output_dir)

#/home/smm/.cache/huggingface/datasets/MathLLMs/MathVision-images
#/home/smm/.cache/huggingface/datasets/MathLLMs/MathVision-images/images/