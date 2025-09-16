# norm full .md 




# norm figs.json
import re, base64, json, os, mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple 
from tqdm import tqdm 





_DIGIT_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")

def _to_ascii_digits(s: str) -> str:
    return s.translate(_DIGIT_TRANS)

def _clean_desc(desc: str) -> str:
    """把'图N为/是/:'去掉，保留后半句。"""
    s = desc.strip()
    m = re.match(r'^图\s*([0-9０-９]+)\s*[:：\s]*(是|为)?\s*', s)
    if m:
        return s[m.end():].strip()
    return s

def _to_path_str(p: Any) -> str | None:
    if not p:
        return None
    if isinstance(p, Path):
        return str(p)
    return str(p)

def _img_to_b64(path_str: str | None,
                include_b64: bool,
                safe_read: bool,
                max_b64_mb: float,
                data_uri: bool) -> str:
    if not include_b64 or not path_str:
        return ""
    try:
        if safe_read and not os.path.exists(path_str):
            return ""
        sz = os.path.getsize(path_str)
        if max_b64_mb and sz > max_b64_mb * 1024 * 1024:
            return ""
        with open(path_str, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        if data_uri:
            mime = mimetypes.guess_type(path_str)[0] or "image/jpeg"
            return f"data:{mime};base64,{b64}"
        return b64
    except Exception:
        return ""

def build_figs_repo(
    figs: Dict[str, Any],
    *,
    include_b64: bool = True,
    include_path: bool = True,  # 
    safe_read: bool = True,
    max_b64_mb: float = 8.0,
    data_uri: bool = False,
) -> Dict[str, Any]:
    """把 figs_MetaDict.json 规范化为可直接入库/前端使用的结构（均可 JSON 序列化）。"""
    repo: Dict[str, Any] = {
        # 摘要图: [path_str | None, base64_str]
        "im_abs": [None, ""],
        # 图号 -> 描述（去掉“图N为/是”）
        "ims_desc": {},
        # 图号 -> 绝对路径（字符串；按开关）
        "ims_absp": {},
        # 图号 -> base64（按开关）
        "ims_bs64": {},
        # 附图标记说明（原始“漂亮字符串”）
        "ims_annos": figs.get("annos_ims", "").strip(),
    }

    # 兼容 im_abs / img_abs
    abs_entry = figs.get("im_abs") or figs.get("img_abs")
    if isinstance(abs_entry, list) and len(abs_entry) >= 2:
        abs_path_str = _to_path_str(abs_entry[1])
        abs_b64 = _img_to_b64(abs_path_str, include_b64, safe_read, max_b64_mb, data_uri)
        repo["im_abs"] = [abs_path_str if include_path else None, abs_b64]

    # 逐个图条目：im_1, im_2, ...
    for k, v in figs.items():
        m = re.match(r'^im[_-]?(\d+)$', k)
        if not m:
            continue
        n = int(_to_ascii_digits(m.group(1)))
        if isinstance(v, list) and len(v) >= 2:
            desc_raw, p = v[0], v[1]
            desc_clean = _clean_desc(str(desc_raw))
            repo["ims_desc"][n] = desc_clean or str(desc_raw).strip()

            p_str = _to_path_str(p)
            if include_path and p_str:
                repo["ims_absp"][n] = p_str
            if include_b64:
                repo["ims_bs64"][n] = _img_to_b64(p_str, include_b64, safe_read, max_b64_mb, data_uri)

    # 兜底：如果没有从 im_n 抽到描述，尝试从 lines_ims 里扫
    if not repo["ims_desc"] and isinstance(figs.get("lines_ims"), list):
        for line in figs["lines_ims"]:
            m2 = re.match(r'^图\s*([0-9０-９]+)\s*', line)
            if not m2:
                continue
            n = int(_to_ascii_digits(m2.group(1)))
            repo["ims_desc"][n] = _clean_desc(line)

    # 若不开 base64/路径，保证字段仍是可序列化的空 dict
    if not include_b64:
        repo["ims_bs64"] = {}
    if not include_path:
        repo["ims_absp"] = {}

    # 可选：把图号按升序整理（JSON keys 仍会是字符串，但顺序更可读）
    repo["ims_desc"] = {k: repo["ims_desc"][k] for k in sorted(repo["ims_desc"])}
    repo["ims_absp"] = {k: repo["ims_absp"][k] for k in sorted(repo["ims_absp"])}
    repo["ims_bs64"]  = {k: repo["ims_bs64"].get(k, "") for k in sorted(repo["ims_desc"])}

    return repo


def figs_norm_pipe(root_dir: Path):
    jsps = list(Path(root_dir).rglob("figs_MetaDict.json"))
    for j in tqdm(jsps):
        with open(j, "r", encoding='utf-8') as f:
            data = json.load(f)
            
            repo = build_figs_repo (data)
        
        tgtp = j.with_name("figs.json")
        with open(tgtp, "w", encoding="utf-8") as fo:
            json.dump(repo, fo, ensure_ascii=False, indent=2)

def figs_norm_bcn(
    root_dir: Path,
    include_b64: bool = True,
    include_path: bool = False,
    safe_read: bool = True,
    max_b64_mb: float = 5.0,
    data_uri: bool = False,
    overwrite: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    扫描 root_dir 下所有 figs_MetaDict.json，将其规范化为新的 figs.json（旁存）。
    - include_b64: 是否写入 base64
    - include_path: 是否写入本地路径
    - safe_read: 读文件前检查存在性
    - max_b64_mb: 单图超过该体积则不转 base64（''）
    - data_uri: base64 是否携带 data: 前缀
    - overwrite: figs.json 已存在时是否覆盖
    - dry_run: 只预览不写文件
    """
    root_dir = Path(root_dir)
    jsps: List[Path] = list(root_dir.rglob("figs_MetaDict.json"))

    done, skipped, errors = 0, 0, 0
    outputs: List[Tuple[Path, Path]] = []

    for j in tqdm(jsps, desc="Normalizing figs_MetaDict.json"):
        try:
            with open(j, "r", encoding="utf-8") as f:
                raw = json.load(f)

            tgtp = j.with_name("figs.json")
            if tgtp.exists() and not overwrite:
                skipped += 1
                continue

            # 规范化
            repo = build_figs_repo(
                raw,
                include_b64=include_b64,
                include_path=include_path,
                safe_read=safe_read,
                max_b64_mb=max_b64_mb,
                data_uri=data_uri,
            )

            if not dry_run:
                with open(tgtp, "w", encoding="utf-8") as fo:
                    json.dump(repo, fo, ensure_ascii=False, indent=2)
            outputs.append((j, tgtp))
            done += 1

        except Exception as e:
            errors += 1
            # 可换成日志系统
            print(f"[ERROR] {j}: {e}")

    return {
        "found": len(jsps),
        "done": done,
        "skipped": skipped,
        "errors": errors,
        "outputs": outputs,
        "options": {
            "include_b64": include_b64,
            "include_path": include_path,
            "safe_read": safe_read,
            "max_b64_mb": max_b64_mb,
            "data_uri": data_uri,
            "overwrite": overwrite,
            "dry_run": dry_run,
        },
    }







if __name__ == "__main__":
    root = r".log/SimplePDF"
    figs_norm_pipe(root)
