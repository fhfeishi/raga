from injector import singleton
from dataclasses import dataclass 
from functools import lru_cache
from typing import List, Sequence, Union, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


import logging 
logger = logging.getLogger(__name__)


# -------------------- 配置 --------------------
rerank_model_config: Dict[str, Any] = {
    "type": "huggingface",                # 目前支持: "huggingface"
    "huggingface_model": "Qwen/Qwen3-Reranker-0.6B",
    # 可选高级参数
    "device": "cpu",                       # None 自动选择；也可显式 "cuda" / "cpu" / "mps"
    "max_length": 512,                    # 交叉编码器常用 256~512
    "batch_size": 16,                     # 根据显存/内存调
    "trust_remote_code": True,            # Qwen 系列建议 True
    "local_files_only": False,            # 离线/镜像环境可设 True
    "dtype": "auto",                      # "auto" / "float16" / "bfloat16" / "float32"
}


# -------------------- 类型别名 --------------------
try:
    from langchain_core.documents import Document as LCDocument
    LC_DOC_TYPE = LCDocument
except Exception:
    LC_DOC_TYPE = None
DocLike = Union[str, Any]  # str 或 LangChain 的 Document（或兼容对象）



# -------------------- 核心实现 --------------------
@dataclass
class HFReranker:
    model_name: str
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: torch.device
    max_length: int = 512
    batch_size: int = 16

    def _to_text(self, d: DocLike) -> str:
        if isinstance(d, str):
            return d
        # 兼容 LangChain Document / 自定义对象
        if LC_DOC_TYPE is not None and isinstance(d, LC_DOC_TYPE):
            return d.page_content
        # 尝试通用属性
        return getattr(d, "page_content", str(d))

    @torch.inference_mode()
    def score(self, query: str, docs: Sequence[DocLike]) -> List[float]:
        """返回每个 doc 的相关性得分（越高越相关）"""
        texts = [self._to_text(d) for d in docs]
        scores: List[float] = []

        self.model.eval()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # ★ 关键：成对输入 & 显式 padding
            pairs = [(query, t) for t in batch]
            enc = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            # enc = {k: v.to(self.device) for k, v in enc.items()}
            enc.pop("token_type_ids", None)  # 有些模型没有
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits  # [B, 1] 或 [B, 2]

            if logits.size(-1) == 1:
                # 回归头（例如部分 bge-reranker）
                batch_scores = logits.squeeze(-1).float().tolist()
            elif logits.size(-1) == 2:
                # 二分类：取“相关”类别概率或 logit
                # 用 softmax 概率更稳定（0~1），也便于跨批比较
                probs = torch.softmax(logits, dim=-1)[..., 1]
                batch_scores = probs.float().tolist()
            else:
                # 兜底：取最后一维均值
                batch_scores = logits.mean(dim=-1).float().tolist()

            scores.extend(batch_scores)

        return scores

    def rerank(
        self,
        query: str,
        docs: Sequence[DocLike],
        top_k: Optional[int] = None,
        with_score: bool = True,
        batch_size: Optional[int] = None, 
        max_length: Optional[int] = None,
    ) -> List[Union[DocLike, Dict[str, Any]]]:
        """按相关性降序返回文档；可选只取 top_k。"""
        if not docs:
            return []
        
        
        # 允许临时覆盖批大小/长度
        old_bs, old_ml = self.batch_size, self.max_length
        if batch_size: self.batch_size = batch_size
        if max_length: self.max_length = max_length
        try:
            scores = self.score(query, docs)
        finally:
            self.batch_size, self.max_length = old_bs, old_ml

        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        
        if top_k is not None:
            order = order[:top_k]

        if with_score:
            return [{"doc": docs[i], "score": float(scores[i])} for i in order]
        else:
            return [docs[i] for i in order]


# -------------------- 工厂函数（仿照 chatmodel 模块） --------------------
def _select_device(cfg_device: Optional[str] = None) -> torch.device:
    if cfg_device:
        return torch.device(cfg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_dtype(dtype_str: str, device: torch.device):
    if dtype_str == "auto":
        if device.type == "cuda":
            return torch.float16
        # Apple MPS 或 CPU 更推荐 bfloat16/float32（具体看模型支持）
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str.lower(), torch.float32)


@lru_cache(maxsize=1)  # 进程级单例，等价于 @singleton 的效果
def rerank_model_createfn(cfg: Dict[str, Any] = tuple(rerank_model_config.items())) -> HFReranker:
    """
    返回一个已经就绪的 HFReranker 实例。
    说明：用 lru_cache 保证多次调用只创建一次（简单单例）。
    如果你在 Injector Module 里用 provider/@singleton，也可以直接调用本函数内部逻辑。
    """
    # lru_cache 不接受可变 dict，故传 tuple(items)；这里转回 dict
    if isinstance(cfg, tuple):
        cfg = dict(cfg)

    rtype = (cfg.get("type") or "huggingface").lower()
    if rtype != "huggingface":
        raise NotImplementedError(f"Reranker type='{rtype}' 暂不支持，仅支持 'huggingface'")

    model_name = cfg["huggingface_model"]
    device = _select_device(cfg.get("device"))
    dtype = _select_dtype(cfg.get("dtype", "auto"), device)

    logger.info(f"[Rerank] Loading HF cross-encoder: {model_name} on {device} (dtype={dtype})")

    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        local_files_only=bool(cfg.get("local_files_only", False)),
    )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        local_files_only=bool(cfg.get("local_files_only", False)),
    )

    # ★ 关键：确保有 pad_token_id
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
            mdl.resize_token_embeddings(len(tok))
    mdl.config.pad_token_id = tok.pad_token_id
    tok.padding_side = "right"

    mdl = mdl.to(device)
    logger.info(f"[Rerank] Ready. pad_token_id={tok.pad_token_id}")

    reranker = HFReranker(
        model_name=model_name,
        tokenizer=tok,
        model=mdl,
        device=device,
        max_length=int(cfg.get("max_length", 512)),
        batch_size=int(cfg.get("batch_size", 16)),
    )
    logger.info("[Rerank] Ready.")
    return reranker

    
    
    



