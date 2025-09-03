# fh_models_download.py
# Python 3.9+ / huggingface_hub(new) / transformers(new)

from __future__ import annotations

import os
import shutil
import fnmatch
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
    HfFolder,
    login,
    constants,
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

StrPath = Union[str, os.PathLike]


# ============================== Cache / ENV ==============================

class HFModelsDownloader_Cache:
    """
    统一管理 Hugging Face 缓存路径与环境变量。
    - 设置/查看 hub 缓存目录（HUGGINGFACE_HUB_CACHE / HF_HOME / TRANSFORMERS_CACHE）
    - 列出/清理缓存（支持通配符）
    """

    def __init__(self, cache_dir: Optional[StrPath] = None) -> None:
        if cache_dir:
            self.set_cache_dir(cache_dir)

    @staticmethod
    def set_cache_dir(cache_dir: StrPath, set_transformers_cache: bool = True) -> Path:
        """
        推荐使用：设置 HUGGINGFACE_HUB_CACHE（新版本优先读取）。
        同时“尽量不覆盖”已有 HF_HOME（保留用户已有设置）。
        """
        cache_dir = Path(cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
        os.environ.setdefault("HF_HOME", str(cache_dir))  # 若你单独想放在 HF_HOME/hub，可自行修改

        if set_transformers_cache:
            os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

        return cache_dir

    @staticmethod
    def describe_env() -> Dict[str, Any]:
        """返回与缓存/登录相关的环境与常量，便于排障。"""
        return {
            "env": {
                "HUGGINGFACE_HUB_CACHE": os.getenv("HUGGINGFACE_HUB_CACHE"),
                "HF_HOME": os.getenv("HF_HOME"),
                "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
                "HF_ENDPOINT": os.getenv("HF_ENDPOINT"),
                "HF_HUB_ENABLE_HF_TRANSFER": os.getenv("HF_HUB_ENABLE_HF_TRANSFER"),
                "HF_HUB_OFFLINE": os.getenv("HF_HUB_OFFLINE"),
            },
            "constants": {
                "HF_HUB_CACHE": str(constants.HF_HUB_CACHE),
                "HF_HOME": str(constants.HF_HOME),
                "DEFAULT_ENDPOINT": constants.DEFAULT_ENDPOINT,
            },
            "token_present": bool(HfFolder.get_token() or os.getenv("HUGGINGFACE_TOKEN")),
        }

    @staticmethod
    def list_cached_repos(limit: int = 200) -> List[str]:
        """
        扫描 hub 缓存目录，粗略列出已缓存的 repo_id（模型/空间/数据集）。
        目录模式：<cache>/(models|datasets|spaces)--org--repo/...
        """
        root = Path(constants.HF_HUB_CACHE)
        results: List[str] = []
        if not root.exists():
            return results

        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name  # 例如 models--org--repo
            if "--" in name:
                parts = name.split("--", 2)
                if len(parts) == 3:
                    repo_type, org, repo = parts
                    repo_type = repo_type.replace("models", "model").replace("spaces", "space")
                    rid = f"{org}/{repo}"
                    results.append(f"{repo_type}:{rid}")
        return results[:limit]

    @staticmethod
    def clear_cache(patterns: Optional[List[str]] = None, dry_run: bool = True) -> List[Path]:
        """
        清理 hub 缓存；默认 dry_run=True 仅预览。
        - 不传 patterns：只清理“.incomplete/.lock”等残片
        - 传 patterns：按通配符匹配 <cache>/* 的顶层目录名
          例如 ["models--*Qwen*", "datasets--*my_corpus*"]
        """
        root = Path(constants.HF_HUB_CACHE)
        to_remove: List[Path] = []
        if not root.exists():
            return to_remove

        if not patterns:
            garbage = [p for p in root.rglob("*") if p.name.endswith(".incomplete") or p.name.endswith(".lock")]
            to_remove.extend(garbage)
        else:
            for child in root.iterdir():
                for pat in patterns:
                    if fnmatch.fnmatch(child.name, pat):
                        to_remove.append(child)

        if not dry_run:
            for p in to_remove:
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink(missing_ok=True)
                    else:
                        shutil.rmtree(p, ignore_errors=True)
                except Exception as e:
                    print(f"[WARN] 删除失败: {p} :: {e}")
        return to_remove


# ============================== AutoLoad (transformers) ==============================

class HFModelsDownloader_AutoLoad:
    """
    使用 transformers.from_pretrained 下载/加载到缓存（支持离线回退）。
    - device_map="auto" 显存/设备自动映射
    - 支持 torch_dtype / trust_remote_code / local_files_only
    """

    def __init__(
        self,
        cache_dir: Optional[StrPath] = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        if cache_dir:
            HFModelsDownloader_Cache.set_cache_dir(cache_dir)
        self.token = token or os.getenv("HUGGINGFACE_TOKEN") or HfFolder.get_token()
        self.trust_remote_code = trust_remote_code

    def load_tokenizer(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        local_files_only: bool = False,
    ):
        return AutoTokenizer.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
            local_files_only=local_files_only,
        )

    def load_causallm(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        local_files_only: bool = False,
        low_cpu_mem_usage: bool = True,
    ):
        if torch_dtype is None:
            if torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else None

        return AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
            low_cpu_mem_usage=low_cpu_mem_usage,
            local_files_only=local_files_only,
        )

    def load_with_offline_fallback(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        try_online_first: bool = True,
    ):
        # 在线优先
        if try_online_first:
            try:
                tok = self.load_tokenizer(repo_id, revision=revision, local_files_only=False)
                mdl = self.load_causallm(
                    repo_id,
                    revision=revision,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    local_files_only=False,
                )
                return tok, mdl
            except Exception as e:
                print(f"[WARN] 在线加载失败，尝试离线回退：{e}")

        # 离线回退
        tok = self.load_tokenizer(repo_id, revision=revision, local_files_only=True)
        mdl = self.load_causallm(
            repo_id,
            revision=revision,
            device_map=device_map,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        return tok, mdl


# ============================== Snapshot (整库下载) ==============================

class HFModelsDownloader_SnapShot:
    """
    使用 huggingface_hub.snapshot_download 进行整库下载（支持 allow/ignore patterns）。
    """

    def __init__(self, cache_dir: Optional[StrPath] = None, token: Optional[str] = None, offline: bool = False) -> None:
        if cache_dir:
            HFModelsDownloader_Cache.set_cache_dir(cache_dir)
        self.cache_dir = cache_dir
        self.token = token or os.getenv("HUGGINGFACE_TOKEN") or HfFolder.get_token()
        if self.token:
            try:
                login(self.token, add_to_git_credential=False)
            except Exception:
                pass
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

    def download_repo(
        self,
        repo_id: str,
        repo_type: Optional[str] = None,   # "model" | "dataset" | "space"
        revision: Optional[str] = None,    # "main" / tag / commit sha
        local_dir: Optional[StrPath] = None,
        allow_patterns: Optional[Union[str, List[str]]] = None,
        ignore_patterns: Optional[Union[str, List[str]]] = None,
        force_download: bool = False,
        local_dir_use_symlinks: Optional[bool] = None,
        max_workers: Optional[int] = None,
        etag_timeout: float = 10.0,
    ) -> Path:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            # cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
            cache_dir=self.cache_dir,
            local_dir=str(local_dir) if local_dir else None,
            local_dir_use_symlinks=local_dir_use_symlinks,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            force_download=force_download,
            max_workers=max_workers,
            etag_timeout=etag_timeout,
            token=self.token,
        )
        return Path(path)


# ============================== HubLoad (单/多文件) ==============================

class HFModelsDownloader_HubLoad:
    """
    使用 huggingface_hub.hf_hub_download 下载单文件或批量文件到缓存。
    """

    def __init__(self, cache_dir: Optional[StrPath] = None, token: Optional[str] = None, offline: bool = False) -> None:
        if cache_dir:
            HFModelsDownloader_Cache.set_cache_dir(cache_dir)
        self.token = token or os.getenv("HUGGINGFACE_TOKEN") or HfFolder.get_token()
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

    def download_file(
        self,
        repo_id: str,
        filename: str,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        etag_timeout: float = 10.0,
    ) -> Path:
        fp = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
            force_download=force_download,
            etag_timeout=etag_timeout,
            token=self.token,
        )
        return Path(fp)

    def download_many_files(
        self,
        repo_id: str,
        filenames: List[str],
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        etag_timeout: float = 10.0,
    ) -> List[Path]:
        out: List[Path] = []
        for name in filenames:
            out.append(
                self.download_file(
                    repo_id=repo_id,
                    filename=name,
                    repo_type=repo_type,
                    revision=revision,
                    force_download=force_download,
                    etag_timeout=etag_timeout,
                )
            )
        return out


# ============================== CLI / Demo ==============================

"""
# huggingface_hub CLI 示例
$ hf download <repo_id> --local-dir <local_dir> --resume-download
"""

if __name__ == '__main__':
    # —— 镜像源（按需启用其中之一） ——
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 默认
    # os.environ['HF_ENDPOINT'] = 'https://mirrors.tuna.tsinghua.edu.cn/hugging-face/'
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    # os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')

    # —— 缓存目录（示例：Windows 本地盘） ——
    # os.environ['HF_HOME'] = str(Path(r'E:\local_models\huggingface\cache'))
    os.environ.setdefault('HF_HOME', str(Path(r'E:\local_models\huggingface\cache')))
    # 或者直接：HFModelsDownloader_Cache.set_cache_dir(r'E:\local_models\huggingface\cache')

    # —— 本地落地目录（用于 snapshot_download 的 local_dir） ——
    # huggingface_local_dir = Path(r'E:\local_models\huggingface\local')
    huggingface_local_dir = Path(r'E:\local_models\huggingface\local')
    huggingface_local_dir.mkdir(parents=True, exist_ok=True)


    
    
