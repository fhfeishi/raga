# struct   full_split.md 


# normal markdown
"""  
-->
# record       # --> metadata  --> for search 
...
abs

# claims       # 看似不是很重要的东西  --> may used in answer
1. ...
2. ...
...

# mannuals     # 细节     --> add for answer


将“专利文本（粗糙 Markdown）”规范化为结构化 Markdown：
- 归并“著录信息”（含摘要）
- 补齐“权利要求书”一级标题
- 归档“说明书”并抽取子段（标题、背景技术、发明/实用新型内容、具体实施方式）
- 忽略“附图说明 / 说明书附图”段（按用户要求）

"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CLAIM_LINE_RE       = re.compile(r"^\s*[（(]?\s*\d+\s*[)）\.．、]\s*")   # 1. / 1、 / 1) / （1）/ 1．
ABS_ANCHOR_RE       = re.compile(r"^\s*#?\s*\(57\)\s*摘要\s*$")          # “(57)摘要”行（容错带/不带 #）
NAME_54_RE          = re.compile(r"^\s*#?\s*\(54\)\s*(?:发明名称|实用新型名称|名称|题名)\s*$")
TYPE_UTILITY_RE     = re.compile(r"\(12\).*?实用新型")
TYPE_INVENTION_RE   = re.compile(r"\(12\).*?发明")
H1_RE               = re.compile(r"^\s*#\s+(?P<title>.+?)\s*$")
ANY_H_RE            = re.compile(r"^\s*#+\s*(?P<title>.+?)\s*$")

# —— 说明书子段识别（含常见变体/同义标题）——
SEC_TECH        = {"技术领域"}
SEC_BG          = {"背景技术"}
SEC_CONTENT_INV = {"发明内容"}                 # 严格按你要求，仅认“发明内容”
SEC_CONTENT_UM  = {"实用新型内容"}              # 严格按你要求，仅认“实用新型内容”
SEC_IMPL        = {"具体实施方式"}
SEC_FIGS        = {"附图说明", "说明书附图"}     # 忽略输出

# 也把“裸标题”（没有 # 的行）当作分段/截断锚点
PLAIN_HEADINGS = SEC_TECH | SEC_BG | SEC_CONTENT_INV | SEC_CONTENT_UM | SEC_IMPL | SEC_FIGS

@dataclass
class Pieces:
    patent_type   : str                  # 'utility' | 'invention' | 'unknown'
    meta_prefix   : List[str]            # 摘要行之前的 (xx) 信息块（已去掉行首#）
    abstract_lines: List[str]            # 摘要正文（不含 "(57) 摘要" 行）
    claim_lines   : List[str]            # 权利要求条款行（整块，含内部空行）
    body_rest     : List[str]            # 余下说明书原始行
    title_text    : Optional[str]        # 标题
    sections      : Dict[str, List[str]] # 说明书分段原文（按识别标题键名收拢）

class PatentMdStruct:
    """
    将 full_split.md 重构并写同目录：full_split_struct.md
    输出结构为：
    # record  著录信息
    ...(著录信息行)
    (57) 摘要
    ...(摘要正文)

    # claims 权利要求书
    1. ...
    2. ...
    ...

    # manuals 说明书 
    标题：
       xxx
    背景技术：
       ...
    发明内容：/实用新型内容：
       ...
    具体实施方式：
       ...
    """
    def __init__(self, markdown_file_path: str | Path):
        self.src = Path(markdown_file_path)
        self.text = self.src.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")

    def __call__(self) -> Path:
        pcs = self._segment(self.text)
        md  = self._compose(pcs)
        out = self.src.with_name("full_split_struct.md")
        out.write_text(md, encoding="utf-8")
        return out

    # ---------------- segmentation ----------------
    def _segment(self, txt: str) -> Pieces:
        lines = txt.split("\n")

        # 专利类型（前 60 行粗判）
        head60 = "\n".join(lines[:60])
        if TYPE_UTILITY_RE.search(head60):
            ptype = "utility"
        elif TYPE_INVENTION_RE.search(head60):
            ptype = "invention"
        else:
            ptype = "unknown"

        # 摘要锚点
        abs_idx = None
        for i, ln in enumerate(lines):
            if ABS_ANCHOR_RE.match(ln.strip()):
                abs_idx = i; break

        meta_prefix, abstract_lines, claim_lines, body_rest = [], [], [], []

        if abs_idx is None:
            # 无“(57) 摘要”标题：把开头信息先收为 meta，直到发现权利要求或“说明书段标题/裸标题”
            cut = 0
            while cut < len(lines) and not CLAIM_LINE_RE.match(lines[cut]) and not self._looks_body_heading_or_plain(lines[cut]):
                meta_prefix.append(self._strip_leading_hash(lines[cut])); cut += 1
            # 权利要求块（无摘要时，同样按“首个编号行→下一个分节标题（含裸标题）”提取）
            c_s, c_e = self._find_claims_block(lines, start=cut)
            if c_s is not None:
                claim_lines = lines[c_s:c_e]
                body_rest   = lines[c_e:]
            else:
                body_rest   = lines[cut:]
        else:
            # 著录信息到摘要标题行为止
            for ln in lines[:abs_idx]:
                meta_prefix.append(self._strip_leading_hash(ln))
            # 摘要正文：从摘要行下一行，直到出现权利要求起点或“说明书段标题/裸标题”
            j = abs_idx + 1
            while j < len(lines) and not CLAIM_LINE_RE.match(lines[j]) and not self._looks_body_heading_or_plain(lines[j]):
                abstract_lines.append(lines[j].rstrip()); j += 1
            # 权利要求整块：从首个编号行开始，到下一个分节标题（包含裸标题）
            c_s, c_e = self._find_claims_block(lines, start=j)
            if c_s is not None:
                claim_lines = lines[c_s:c_e]
                body_rest   = lines[c_e:]
            else:
                body_rest   = lines[j:]

        sections   = self._extract_body_sections(body_rest)
        title_text = self._extract_title(lines, sections)

        return Pieces(
            patent_type=ptype,
            meta_prefix=self._trim(meta_prefix),
            abstract_lines=self._trim(abstract_lines),
            claim_lines=self._trim_trailing_blank(claim_lines),  # 保留内部空行，去掉尾部空行
            body_rest=body_rest,
            title_text=title_text,
            sections=sections,
        )

    def _find_claims_block(self, lines: List[str], start: int) -> Tuple[Optional[int], Optional[int]]:
        """
        从 start 开始寻找首个权利要求编号行（CLAIM_LINE_RE），
        返回 [c_start, c_end) 区间，其中 c_end 为后续遇到的首个“Markdown 标题(#开头) 或 裸标题行(技术领域/背景技术/发明内容/实用新型内容/具体实施方式/附图说明/说明书附图)”。
        若未找到编号行，返回 (None, None)。
        """
        n = len(lines)
        c_start = None
        i = start
        while i < n:
            if CLAIM_LINE_RE.match(lines[i]):
                c_start = i
                break
            # 如果在遇到编号前先撞到了“分节标题/裸标题”，说明没有权利要求
            if self._looks_body_heading_or_plain(lines[i]):
                return None, None
            i += 1
        if c_start is None:
            return None, None

        # 往后一直吃，直到遇到分节标题/裸标题
        j = c_start + 1
        while j < n and not self._looks_body_heading_or_plain(lines[j]):
            j += 1
        return c_start, j

    # ---------------- compose ----------------
    def _compose(self, p: Pieces) -> str:
        out: List[str] = []

        # record：著录信息 + 摘要
        out.append("# 著录信息")     
        out.extend(self._trim(p.meta_prefix))
        out.append("")
        out.append("(57) 摘要")
        out.extend(p.abstract_lines)
        out.append("")

        # claims：整块原样放入
        out.append("# 权力要求书")
        out.extend(p.claim_lines)
        out.append("")

        # manuals：说明书（四小节）
        out.append("# 说明书")

        # 标题
        out.append("标题：")
        if p.title_text:
            out.append(f"   {p.title_text}")
        out.append("")

        # 背景技术
        out.append("背景技术：")
        bg = p.sections.get("背景技术", [])
        out.extend(bg)
        out.append("")

        # 发明内容 / 实用新型内容（二选一标题）
        content_title, content_lines = self._pick_content_block(p)
        out.append(f"{content_title}：")
        out.extend(content_lines)
        out.append("")

        # 具体实施方式
        out.append("具体实施方式：")
        impl = []
        for k in SEC_IMPL:
            if k in p.sections:
                impl = p.sections[k]; break
        out.extend(impl)
        out.append("")

        return "\n".join(out)

    # ---------------- helpers ----------------
    def _pick_content_block(self, p: Pieces) -> Tuple[str, List[str]]:
        if "发明内容" in p.sections:
            return "发明内容", p.sections["发明内容"]
        if "实用新型内容" in p.sections:
            return "实用新型内容", p.sections["实用新型内容"]
        # 如果二者都没有，则按类型兜底给空段
        return ("发明内容" if p.patent_type == "invention" else "实用新型内容"), []

    @staticmethod
    def _strip_leading_hash(s: str) -> str:
        s2 = s.lstrip()
        if s2.startswith("# "):  # 只把顶层 # 去掉，保持行文本
            return s2[2:].strip()
        return s.rstrip()

    @staticmethod
    def _trim(lines: List[str]) -> List[str]:
        i, j = 0, len(lines)
        while i < j and not lines[i].strip():
            i += 1
        while j > i and not lines[j - 1].strip():
            j -= 1
        return [ln.rstrip() for ln in lines[i:j]]

    @staticmethod
    def _trim_trailing_blank(lines: List[str]) -> List[str]:
        # 保留内部空行，只去掉末尾空白
        j = len(lines)
        while j > 0 and not lines[j - 1].strip():
            j -= 1
        return [ln.rstrip() for ln in lines[:j]]

    @staticmethod
    def _title_clean(t: str) -> str:
        return re.sub(r"\s+", " ", t).strip()

    def _extract_title(self, lines: List[str], sections: Dict[str, List[str]]) -> Optional[str]:
        # 优先 (54) 名称 下一行非 # 的文本
        idx54 = None
        for i, ln in enumerate(lines[:200]):
            if NAME_54_RE.match(ln):
                idx54 = i; break
        if idx54 is not None:
            j = idx54 + 1
            while j < len(lines):
                cand = lines[j].strip()
                if cand and (not ANY_H_RE.match(cand)):
                    return self._title_clean(cand)
                j += 1
        # 次选：第一个不是通用小节名的一级标题
        known = PLAIN_HEADINGS
        for ln in lines:
            m = H1_RE.match(ln)
            if m:
                t = m.group("title").strip()
                if t not in known and not t.startswith("("):
                    return self._title_clean(t)
        # 再兜底：任一不属于已知集合的分节名
        for k in sections.keys():
            if k not in known:
                return self._title_clean(k)
        return None

    def _looks_body_heading_or_plain(self, line: str) -> bool:
        """既识别以 # 开头的 Markdown 标题，也识别没有 # 的‘裸标题行’。"""
        s = line.strip()
        m = ANY_H_RE.match(s)
        if m:
            t = m.group("title").strip()
            return t in (PLAIN_HEADINGS)
        # 裸标题：整行等于这些关键字
        return s in PLAIN_HEADINGS

    def _extract_body_sections(self, body_lines: List[str]) -> Dict[str, List[str]]:
        """
        从 body_lines（摘要与权利要求书之后的剩余部分）中，按一级 # 或裸标题切分并抽取：
        - 技术领域
        - 背景技术
        - 发明内容 / 实用新型内容
        - 具体实施方式
        忽略“附图说明/说明书附图”。
        """
        secs: Dict[str, List[str]] = {}
        if not body_lines:
            return secs

        # 定位各块起点：既支持 "# 标题" 也支持裸标题
        indices: List[Tuple[str, int]] = []  # (标题, start_line_idx_of_block)
        for i, ln in enumerate(body_lines):
            s = ln.strip()
            m = ANY_H_RE.match(s)
            if m:
                title = m.group("title").strip()
                if title in PLAIN_HEADINGS:
                    indices.append((title, i))
            elif s in PLAIN_HEADINGS:
                indices.append((s, i))

        # 若完全没有标题，就把剩余整体作为“具体实施方式”
        if not indices:
            content = [x.rstrip() for x in body_lines]
            content = self._trim(content)
            if content:
                secs["具体实施方式"] = content
            return secs

        # 收尾标记
        indices.append(("#END#", len(body_lines)))

        # 提取各段
        for (title, s), (_, e) in zip(indices, indices[1:]):
            if title in SEC_FIGS:
                continue  # 忽略附图说明类
            content = [ln.rstrip() for ln in body_lines[s + 1:e]]
            content = self._trim(content)
            if content:
                secs[title] = content

        return secs    
    
# ---- 示例运行（可注释）----
if __name__ == "__main__":
    
    from tqdm import tqdm

    root_dir = Path.cwd().parent.parent.parent / "./.log" / "SimplePDF"
    assert Path(root_dir).is_file()
    tft_mds = list(Path(root_dir).rglob("full_split.md"))
    for md in tqdm(tft_mds):
        demo = str(md)
        outp = PatentMdStruct(demo)()
        print("写入：", outp)
