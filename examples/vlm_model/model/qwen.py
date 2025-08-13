from modelscope import snapshot_download, logging
import os

# 开启 ModelScope debug 日志
logging.set_verbosity(logging.DEBUG)

# 替换为你想要的模型 ID
model_id = 'Qwen/Qwen2.5-VL-7B-Instruct'

print(f"[DEBUG] 开始下载模型：{model_id}")

model_dir = snapshot_download(model_id=model_id)

print(f"[DEBUG] 模型已下载到：{model_dir}")

# 列出下载目录的文件结构（最多两级）
for root, dirs, files in os.walk(model_dir):
    level = root.replace(model_dir, '').count(os.sep)
    if level < 2:
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
