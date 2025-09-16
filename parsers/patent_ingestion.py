# ingestion     process   retrieve 

# embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embedding = HuggingFaceEmbedding(
    # cache
    model_name="Qwen/Qwen3-Embedding-0.6B",
    max_length=1024,
    trust_remote_code=True,
    model_kwargs={"local_files_only": True},   # 允许联网 False , 禁止联网 True
    device='cpu',
)















