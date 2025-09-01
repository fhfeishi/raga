import os, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM



import sys, transformers, huggingface_hub, sentence_transformers
print("py:", sys.executable)
print("tf:", transformers.__version__, "hub:", huggingface_hub.__version__, "st:", sentence_transformers.__version__)
print("HF_HOME:", os.getenv("HF_HOME"))
print("HF_ENDPOINT:", os.getenv("HF_ENDPOINT"))


# —— 离线&缓存环境（Windows 下也OK）——
os.environ["HF_HOME"] = r"E:\local_models\huggingface\cache"
# os.environ["HF_HUB_OFFLINE"] = "0"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 关闭 symlink 警告
os.environ["HF_ENDPOINT"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face/"


name_or_path = "Qwen/Qwen3-Embedding-0.6B"  # 也可换成本地目录（见方案2）
device = "cpu"  # 或 "cuda:0"（如果有GPU）

# 第一次若缓存不全，把 local_files_only 改为 False 联网补齐一次
tokenizer = AutoTokenizer.from_pretrained(
    name_or_path, trust_remote_code=True, local_files_only=True,
    cache_dir=os.environ["HF_HOME"],
)
model = AutoModelForCausalLM.from_pretrained(
    name_or_path, trust_remote_code=True, local_files_only=True,
    cache_dir=os.environ["HF_HOME"],
).to(device).eval()



