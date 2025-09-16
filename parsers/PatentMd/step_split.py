# split full.md 
# -> full_split.md (文本信息)       figs.json  （结构化的图片信息）

"""  full_split.md (文本信息)       figs.json  （结构化的图片信息）

full_split.md (文本信息)  --->  结构还有待优化 step_struct.py

figs.json  （结构化的图片信息）
{
  "im_abs": [ "<abs_path_or_null>", "<base64_or_empty>" ],
  "ims_desc": { "1": "……", "2": "……" },
  "ims_absp": { "1": "<abs_path>", "2": "<abs_path>" },
  "ims_bs64": { "1": "<b64>", "2": "<b64>" },
  "ims_annos": "图中…；…"
}
"""
 

import os
import re
import json
import base64
import mimetypes
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path

# -------------------- 正则 --------------------
IMG_MD_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)', re.IGNORECASE)
HEADER_RE = re.compile(r'^\s*#{1,6}\s*(?P<title>.+?)\s*$', re.MULTILINE)
ABSTRACT_HEADER_RE = re.compile(r'^\s*#\s*(?:\(57\))?\s*摘要\s*$', re.MULTILINE)
FIGDESC_HEADER_RE = re.compile(r'^\s*#\s*附图说明\s*$', re.MULTILINE)
CN_INDEX_TAG_RE = re.compile(r'\[\s*\d{3,}\s*\]')  # [0017] 这类编号

# -------------------- 工具函数：构建 figs.json --------------------
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

def _to_path_str(p: Any) -> Optional[str]:
    if not p:
        return None
    return str(p)

def _img_to_b64(path_str: Optional[str],
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
    include_path: bool = False,
    safe_read: bool = True,
    max_b64_mb: float = 8.0,
    data_uri: bool = False,
) -> Dict[str, Any]:
    """把 figs_MetaDict 规范化为可直接入库/前端使用的结构（均可 JSON 序列化）。"""
    repo: Dict[str, Any] = {
        # 摘要图: [path_str | None, base64_str]
        "im_abs": [None, ""],
        # 图号 -> 描述（去掉“图N为/是”）
        "ims_desc": {},
        # 图号 -> 绝对路径（字符串；按开关）
        "ims_absp": {},
        # 图号 -> base64（按开关）
        "ims_bs64": {},
        # 附图标记说明（“漂亮字符串”/原始）
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

    # 整理顺序
    repo["ims_desc"] = {k: repo["ims_desc"][k] for k in sorted(repo["ims_desc"])}
    repo["ims_absp"] = {k: repo["ims_absp"][k] for k in sorted(repo["ims_absp"])}
    repo["ims_bs64"]  = {k: repo["ims_bs64"].get(k, "") for k in sorted(repo["ims_desc"])}

    return repo

# -------------------- 主类：一次性产出 full_split.md + figs.json --------------------
class PatentMdSplit:
    """
    用法：
        sp = PatentMdSplit(
            mdf=".../full.md",
            include_b64=True,
            include_path=False,
            max_b64_mb=6.0,
            data_uri=False,
            write_meta_raw=True,   # 是否旁存 figs_MetaDict.json
        )
        out_md, out_figs = sp()   # 返回输出文件路径
    """
    def __init__(
        self,
        mdf: str | Path,
        *,
        include_b64: bool = True,
        include_path: bool = False,
        safe_read: bool = True,
        max_b64_mb: float = 8.0,
        data_uri: bool = False,
        write_meta_raw: bool = False,
    ) -> None:
        self.mdf      : Path = Path(mdf)
        self.fdir     : Path = self.mdf.parent.resolve()
        self.imdir    : Path = self.fdir / "images"
        # pdf 可选
        self.pdff     : Optional[Path] = next(self.fdir.rglob('*_origin.pdf'), None)

        # 可选校验
        if not self.imdir.is_dir():
            # 不强制；有的文档可能无图
            pass

        self.text: str = self.mdf.read_text(encoding='utf-8', errors="ignore")

        # 中间产物（原始 figs 元信息）
        self.figs_info: Dict[str, Any] = {}

        # 输出控制
        self.include_b64 = include_b64
        self.include_path = include_path
        self.safe_read = safe_read
        self.max_b64_mb = max_b64_mb
        self.data_uri = data_uri
        self.write_meta_raw = write_meta_raw

    def __call__(self) -> Tuple[Path, Path]:
        return self.pipeline()

    # -------------------- pipeline --------------------
    def pipeline(self) -> Tuple[Path, Path]:
        raw_text = self.text

        # 先做一次全局扫描，记录所有图片以及“图片后一行的图号 caption”
        all_imgs = list(self._scan_all_images_with_captions(raw_text))

        # 1) 摘要图抽取并删除
        content = self._extract_abstract_image_and_strip(raw_text)

        # 2) 附图说明/附图标记说明抽取，并删除整节
        fig_text_items, fig_annos_str, content = self._extract_and_strip_fig_section(content)

        if fig_text_items:
            self.figs_info["lines_ims"] = fig_text_items  # List[str]
        if fig_annos_str:
            self.figs_info["annos_ims"] = fig_annos_str   # str

        # 3) 文末残留的“图片引用 + 图x” 堆叠删除
        content = self._strip_tail_image_blocks(content)

        # 4) 依据 “图号→路径” 配对 + “附图说明里的描述”，生成 im_1/im_2…
        self._build_im_n_entries(all_imgs, fig_text_items)

        # 5) 落盘：纯文本 + figs.json（可选同时写 figs_MetaDict.json）
        out_md, out_figs = self._write_down(content)

        return out_md, out_figs

    # -------------------- 工具函数 --------------------
    def _section_span(self, text: str, header_re: re.Pattern, next_header_re: re.Pattern = HEADER_RE) -> Tuple[int, int]:
        m = header_re.search(text)
        if not m:
            return -1, -1
        start = m.start()
        m2 = next_header_re.search(text, pos=m.end())
        end = m2.start() if m2 else len(text)
        return start, end

    def _scan_all_images_with_captions(self, text: str):
        """
        扫描整篇 md，返回迭代器： {alt, rel, abs, span, fig_no}
        - fig_no: 紧跟图片后若出现“图N …”则返回 N（int），否则 None
        """
        for m in IMG_MD_RE.finditer(text):
            alt = (m.group("alt") or "").strip()
            rel = (m.group("path") or "").strip()
            abs_path = str((self.fdir / rel).resolve())
            # 找图片后不远处的“图N …”
            lookahead_end = min(len(text), m.end() + 400)
            next_chunk = text[m.end():lookahead_end]
            cap = re.search(r'^\s*图\s*([0-9０-９]+)\b', next_chunk, re.MULTILINE)
            fig_no = None
            if cap:
                num_str = cap.group(1)
                try:
                    fig_no = int(self._to_halfwidth_digits(num_str))
                except Exception:
                    fig_no = None
            yield {
                "alt": alt,
                "rel": rel,
                "abs": abs_path,
                "span": (m.start(), m.end()),
                "fig_no": fig_no,
            }

    def _extract_abstract_image_and_strip(self, text: str) -> str:
        """
        在“摘要”段落内寻找第一张图片作为摘要图：
          - figs_info["im_abs"] = ["摘要图", abs_path]
          - 删除该图片引用行
        """
        s, e = self._section_span(text, ABSTRACT_HEADER_RE)
        if s == -1:
            return text

        sub = text[s:e]
        img = IMG_MD_RE.search(sub)
        if not img:
            return text

        rel = img.group("path").strip()
        abs_path = str((self.fdir / rel).resolve())
        self.figs_info["im_abs"] = ["摘要图", abs_path]

        # 删除图片这一整行
        line_start = text.rfind("\n", 0, s + img.start()) + 1
        line_end = text.find("\n", s + img.end())
        if line_end == -1:
            line_end = len(text)
        new_text = text[:line_start] + text[line_end + 1:]
        return new_text

    def _extract_and_strip_fig_section(self, text: str) -> Tuple[List[str], str, str]:
        """
        提取《附图说明》整节：
          - text_ims: ["图1为xxx", "图2是xxx", ...]
          - fig_annos_str: 以“最后一条图描述”为锚点，截其后的“图中标记说明”，做轻度美化
        然后把该整节从 md 中删除。
        """
        s, e = self._section_span(text, FIGDESC_HEADER_RE)
        if s == -1:
            return [], "", text

        section = text[s:e]

        # 去除 [0017] 之类编号以降低干扰（clean 用于提“图n为/是…”；原始 section 保留给锚点匹配）
        clean = CN_INDEX_TAG_RE.sub(" ", section)
        clean = re.sub(r'^\s*#?\s*附图说明\s*', '', clean, flags=re.MULTILINE).strip()
        mfirst = re.search(r'图\s*[0-9０-９]', clean)
        if mfirst:
            clean = clean[mfirst.start():]

        # 1) 图描述行
        text_ims: List[str] = []
        for seg in re.split(r'[；;。]\s*', clean):
            seg = seg.strip()
            if not seg:
                continue
            m = re.match(r'^图\s*([0-9０-９]+)\s*(.*)$', seg)
            if m:
                n = self._to_halfwidth_digits(m.group(1))
                rest = m.group(2).strip(" ：:为是，,")
                if rest:
                    text_ims.append(f"图{n}{('为' if not rest.startswith(('为', '是', ':', '：')) else '')}{rest}")
                else:
                    text_ims.append(f"图{n}")

        # 2) 以最后一条图描述为锚点，抽“图中标记说明”
        fig_annos_str = ""
        if text_ims:
            last_line = text_ims[-1]
            mm = re.match(r'^图\s*([0-9]+)\s*(.*)$', self._to_halfwidth_digits(last_line))
            if mm:
                last_no = mm.group(1)
                core_desc = (mm.group(2) or "")
                core_desc = core_desc.lstrip("为是：:,， ").strip()
                anchor_re = re.compile(
                    r'(?:\[\s*\d{3,}\s*\]\s*)?图\s*' + re.escape(last_no) +
                    r'\s*(?:为|是|:|：)?\s*' +
                    re.escape(core_desc).replace(r'\ ', r'\s*') +
                    r'[^。；;\n]*[。；;]?',
                    re.S
                )
                last_match = None
                for m in anchor_re.finditer(section):
                    last_match = m
                anchor_pos = last_match.end() if last_match else None
                tail = section[anchor_pos:] if anchor_pos is not None else ""
                fig_annos_str = self._beautify_annos_str(tail)

        # 删除整节《附图说明》
        new_text = text[:s] + text[e:]
        return text_ims, fig_annos_str, new_text

    def _beautify_annos_str(self, s: str) -> str:
        """轻度美化：去编号、合并空白、统一中英文标点、连字符去空格，迭代收敛。"""
        if not s:
            return ""
        def _pass(x: str) -> str:
            x = re.sub(r'\[\s*\d{3,}\s*\]', '', x)         # 去 [0031]
            x = re.sub(r'[ \t\r\n]+', ' ', x).strip()      # 合并空白
            # 统一标点
            x = re.sub(r'\s*[:：]\s*', '：', x)
            x = re.sub(r'\s*[,，]\s*', '，', x)
            x = re.sub(r'\s*[;；]\s*', '；', x)
            # 连字符去空格（2- 1 -> 2-1）
            x = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', x)
            # 多余标点收敛
            x = re.sub(r'，{2,}', '，', x)
            x = re.sub(r'；{2,}', '；', x)
            x = x.strip('：，；。 ')
            x = x.rstrip('。；;')
            return x
        prev, cur = None, s
        for _ in range(4):
            prev, cur = cur, _pass(cur)
            if cur == prev:
                break
        return cur

    def _strip_tail_image_blocks(self, text: str) -> str:
        """移除文末连续的 “图片引用 + （可选）图x说明行” 的堆叠。"""
        tail_re = re.compile(
            r'(?:\s*'
            r'!\[[^\]]*\]\([^)]+\)\s*'
            r'(?:\n\s*图[^\n]*\s*)?'
            r')+\s*$',
            re.S
        )
        return tail_re.sub("", text)

    def _build_im_n_entries(self, all_imgs: List[Dict[str, Any]], text_ims: List[str]) -> None:
        """
        组合 im_1/im_2…：
          - 优先用 “图片后紧邻的图号” 来配对路径
          - 其次按出现顺序回退
          - 文本描述优先取 text_ims 里的 “图n为…”；没有就用 alt 或“图n”
        """
        # 1) 把摘要图从候选里排除（已经作为 im_abs）
        abs_img_path = None
        if "im_abs" in self.figs_info and isinstance(self.figs_info["im_abs"], list) and len(self.figs_info["im_abs"]) >= 2:
            abs_img_path = self.figs_info["im_abs"][1]
        candidate_imgs = [it for it in all_imgs if it["abs"] != abs_img_path]

        # 2) 解析 text_ims -> n -> 描述（保留原始“图n为…”句式，后续 build_figs_repo 会清洗）
        desc_map: Dict[int, str] = {}
        for t in text_ims or []:
            m = re.match(r'^图\s*([0-9０-９]+)\s*(.*)$', t.strip())
            if m:
                n = int(self._to_halfwidth_digits(m.group(1)))
                desc = m.group(2).lstrip("：:为是，, ")
                desc_map[n] = ("图%d为%s" % (n, desc)) if desc else ("图%d" % n)

        # 3) 先用“图片后紧邻的图号”精确配对
        no_to_path: Dict[int, str] = {}
        used_idxs = set()
        for idx, it in enumerate(candidate_imgs):
            if it["fig_no"] is not None:
                n = it["fig_no"]
                if n not in no_to_path:
                    no_to_path[n] = it["abs"]
                    used_idxs.add(idx)

        # 4) 对没有图号的图片，按顺序补充到还没有路径的“图n”
        expected_ns = sorted(desc_map.keys()) if desc_map else list(range(1, len(candidate_imgs) + 1))
        rest_imgs = [it for i, it in enumerate(candidate_imgs) if i not in used_idxs]

        j = 0
        for n in expected_ns:
            if n not in no_to_path and j < len(rest_imgs):
                no_to_path[n] = rest_imgs[j]["abs"]
                j += 1

        # 5) 根据 no_to_path 填充 figs_info 的 im_n
        for n in sorted(no_to_path.keys()):
            key = f"im_{n}"
            desc = desc_map.get(n)
            if not desc:
                # 退化用 alt / 或者“图n”
                alt = ""
                for it in candidate_imgs:
                    if it["abs"] == no_to_path[n]:
                        alt = it.get("alt") or ""
                        break
                desc = alt.strip() or f"图{n}"
            self.figs_info[key] = [desc, no_to_path[n]]

    @staticmethod
    def _to_halfwidth_digits(s: str) -> str:
        return s.translate(str.maketrans("０１２３４５６７８９", "0123456789")).strip()

    # -------------------- 落盘 --------------------
    def _write_down(self, content: str) -> Tuple[Path, Path]:
        in_name = Path(self.mdf).stem
        out_md = self.fdir / f"{in_name}_split.md"
        out_md.write_text(content, encoding="utf-8")

        # 可选旁存原始 figs_MetaDict.json（排查用）
        if self.write_meta_raw:
            meta_path = self.fdir / "figs_MetaDict.json"
            with open(meta_path, "w", encoding='utf-8') as fj:
                json.dump(self.figs_info, fj, ensure_ascii=False, indent=2)

        # 规范化 -> figs.json
        repo = build_figs_repo(
            self.figs_info,
            include_b64=self.include_b64,
            include_path=self.include_path,
            safe_read=self.safe_read,
            max_b64_mb=self.max_b64_mb,
            data_uri=self.data_uri,
        )
        out_figs = self.fdir / "figs.json"
        with open(out_figs, "w", encoding="utf-8") as fo:
            json.dump(repo, fo, ensure_ascii=False, indent=2)

        return out_md, out_figs


# -------------------- 批量运行（可选） --------------------
if __name__ == '__main__':
    from tqdm import tqdm

    root_dir = Path(__file__).parent.parent.parent / "./.log" / "SimplePDF"
    print(str(root_dir))
    tft_mds = list(Path(root_dir).rglob("*/full.md"))
    for md in tqdm(tft_mds):
        md = str(md)
        sp = PatentMdSplit(
            md,
            include_b64=False,   # 如需瘦身可改为 False
            include_path=True, # 不暴露本地路径时设 False
            max_b64_mb=6.0,     # image超过6M就不转base64
            data_uri=False,
            write_meta_raw=True, #
        )
        sp()











