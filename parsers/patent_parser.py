# patent_parser.py

from pydantic import BaseModel
from collections import OrderedDict, defaultdict 
from typing import Dict, Tuple, List , Any
from pathlib import Path 
import json, re  

# re
IMG_MD_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)', re.IGNORECASE)
HEADER_RE = re.compile(r'^\s*#{1,6}\s*(?P<title>.+?)\s*$', re.MULTILINE)
ABSTRACT_HEADER_RE = re.compile(r'^\s*#\s*(?:\(57\))?\s*摘要\s*$', re.MULTILINE)
FIGDESC_HEADER_RE = re.compile(r'^\s*#\s*附图说明\s*$', re.MULTILINE)
CN_INDEX_TAG_RE = re.compile(r'\[\s*\d{3,}\s*\]')  # [0017] 这类编号
# FIG_LABELS_KEY_RE = re.compile(r'附图标记说明')
# FIG_NO_LINE_RE = re.compile(r'^\s*图\s*([0-9０-９]+)\s*([：:为是，].*)?$', re.MULTILINE)  # “图1为…”，“图2：…”


# step1  : extract image infomation  -->  full_filtered.md  figs.MetaDict.json
# setp2  : link  <text---image>     and show image with image-info     # .todo
# step3-1 发明专利      : struct text   <著录信息（pdf首页 关键词条、摘要）、权利要求书、说明书（技术领域、背景技术、发明内容、附图说明<figs_MetaDict.json>、具体实施方式）>
# step3-2 实用新型专利  : struct text   <著录信息（pdf首页 关键词条、摘要）、权利要求书、说明书（技术领域、背景技术、实用新型内容、附图说明<figs_MetaDict.json>、具体实施方式）>

class PatentParser:
    
    def __init__(self, mdf: str) -> Any:
        self.mdf      : Path = Path(mdf) 
        self.fdir     : Path = self.mdf.parent.resolve()
        self.imdir    : Path = self.fdir / "images"
        self.pdff     : Path = next(self.fdir.rglob('*_origin.pdf'), None)
        self.figs_info: Dict[str, List[str]] = defaultdict(List)
        assert self.imdir.is_dir() is True
        assert self.pdff.is_file() is True
        self.text = self.mdf.read_text(encoding='utf-8', errors="ignore")
        
    def __call__(self):
        return self.pipeline()
    
    
    def pipeline(self) -> Path:
        raw_text = self.text

        # 先做一次全局扫描，记录所有图片以及“图片后一行的图号 caption”
        all_imgs = list(self._scan_all_images_with_captions(raw_text))

        # 1) 摘要图抽取并删除
        content = self._extract_abstract_image_and_strip(raw_text)

        # 2) 附图说明/附图标记说明抽取，并删除整节
        fig_text_items, fig_annos, content = self._extract_and_strip_fig_section(content)

        if fig_text_items:
            self.figs_info["lines_ims"] = fig_text_items  # List[str]  图的一行描述。图x是xxxx
        if fig_annos:
            self.figs_info["annos_ims"] = fig_annos  # str  图中（数字）标记的含义，一长串字符串
        print(fig_annos)
        
        # 3) 文末残留的“图片引用 + 图x” 块删除（不影响我们已经拿到的路径/配对）
        content = self._strip_tail_image_blocks(content)

        # 4) 依据“图号→路径”配对 + “附图说明里的描述”，生成 im_1/im_2…
        self._build_im_n_entries(all_imgs, fig_text_items)

        # 写入
        self._write_down(content)
        
        print(f"{self.figs_info = }")
        return 

    # -------------------- 工具函数 --------------------
    def _section_span(self, text: str, header_re: re.Pattern, next_header_re: re.Pattern = HEADER_RE) -> Tuple[int, int]:
        """
        返回某个标题（如“摘要”“附图说明”）所在的 [start, end) 区间（end 为下一个标题或文本末尾）。
        找不到则返回 (-1, -1)。
        """
        m = header_re.search(text)
        if not m:
            return -1, -1
        start = m.start()
        # 从匹配位置往后找“下一个标题”
        m2 = next_header_re.search(text, pos=m.end())
        end = m2.start() if m2 else len(text)
        return start, end

    def _scan_all_images_with_captions(self, text: str):
        """
        扫描整篇 md，返回迭代器： (alt, rel_path, abs_path, img_span, following_caption_no)
        - following_caption_no: 紧跟图片后若出现“图N …”则返回 N（int），否则 None
        """
        for m in IMG_MD_RE.finditer(text):
            alt = (m.group("alt") or "").strip()
            rel = (m.group("path") or "").strip()
            abs_path = str((self.fdir / rel).resolve())
            # 找图片后不远处的“图N …”（最多看后面 ~200 字符或到下一张图）
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
        # 找到整行边界后替换为空
        line_start = text.rfind("\n", 0, s + img.start()) + 1
        line_end = text.find("\n", s + img.end())
        if line_end == -1:
            line_end = len(text)
        new_text = text[:line_start] + text[line_end + 1:]
        return new_text

    def _extract_and_strip_fig_section(self, text: str) -> Tuple[List[str], Dict[str, str], str]:
        """
        提取《附图说明》整节：
          - text_ims: ["图1为xxx", "图2是xxx", ...]
          - fig_annos_str: []   <-  最后一个 [xxxx]之后的字符串（美化一下其中字符）
        然后把该整节从 md 中删除。
        """
        s, e = self._section_span(text, FIGDESC_HEADER_RE)
        if s == -1:
            return [], {}, text

        section = text[s:e]

        # 去除 [0017] 之类编号以降低干扰
        clean = CN_INDEX_TAG_RE.sub(" ", section)
        # 去掉行首的 “附图说明” 标题（兼容是否带 #）
        clean = re.sub(r'^\s*#?\s*附图说明\s*', '', clean, flags=re.MULTILINE).strip()
        # 把文本从第一个“图<数字>”的位置切起
        mfirst = re.search(r'图\s*[0-9０-９]', clean)
        if mfirst:
            clean = clean[mfirst.start():]            

        # 1) 提图描述行：图N为/是…
        text_ims: List[str] = []
        # 为了尽可能拿全，用句号/分号分割后逐条匹配
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
                # text_ims_lastLineStr = 
        
        # 2) 附图标记说明：
        # fig_annos_str：用  last_line 锚点 + 美化 —— #
        last_line = text_ims[-1] 
        fig_annos_str = ""
        # tail = str(section).split(str(last_line))[1]
        mm = re.match(r'^图\s*([0-9]+)\s*(.*)$', last_line)
        anchor_pos = None
        last_no = mm.group(1)
        core_desc = (mm.group(2) or "")
        core_desc = core_desc.lstrip("为是：:,， ").strip()
        # 用原始 section 做匹配（保留原有 [00xx] 与换行），匹配 “图<no> (为|是|:|：)? <核心描述> … 到句号/分号为止”
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
        if last_match:
            anchor_pos = last_match.end()
        tail = section[anchor_pos:]
        fig_annos_str = self._beautify_annos_str(tail)

        # 删除整节《附图说明》
        new_text = text[:s] + text[e:]
        return text_ims, fig_annos_str, new_text

    def _beautify_annos_str(self, s: str) -> str:
        """轻度美化：去编号、合并空白、统一中英文标点、连字符去空格，直到稳定。"""
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
        for _ in range(4):  # 简单“递归”：最多迭代几次直到稳定
            prev, cur = cur, _pass(cur)
            if cur == prev:
                break
        return cur
    
    def _strip_tail_image_blocks(self, text: str) -> str:
        """
        移除文末连续的 “图片引用 + （可选）图x说明行” 的堆叠。
        仅作用于末尾，不影响正文中间的插图。
        """
        tail_re = re.compile(
            r'(?:\s*'                       # 前导空白
            r'!\[[^\]]*\]\([^)]+\)\s*'      # 图片
            r'(?:\n\s*图[^\n]*\s*)?'         # 可选下一行的“图x …”
            r')+\s*$',                       # 可以重复多次直到末尾
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
        # 1) 把摘要图从候选里排除（已经作为 img_abs）
        abs_img_path = None
        if "im_abs" in self.figs_info:
            abs_img_path = self.figs_info["im_abs"][1]
        candidate_imgs = [it for it in all_imgs if it["abs"] != abs_img_path]

        # 2) 解析 text_ims -> n -> 描述
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
                if n not in no_to_path:  # 只取第一张
                    no_to_path[n] = it["abs"]
                    used_idxs.add(idx)

        # 4) 对没有图号的图片，按顺序补充到还没有路径的“图n”
        #    如果 text_ims 给了 n 列表，就以它为参考；否则就从 1..len(candidate_imgs)
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
                # 尝试找对应图片的 alt
                for it in candidate_imgs:
                    if it["abs"] == no_to_path[n]:
                        alt = it.get("alt") or ""
                        break
                desc = alt.strip() or f"图{n}"
            self.figs_info[key] = [desc, no_to_path[n]]

    @staticmethod
    def _to_halfwidth_digits(s: str) -> str:
        """
        把全角数字转半角，同时去空白。
        """
        trans = str.maketrans("０１２３４５６７８９", "0123456789")
        return s.translate(trans).strip()

    # -------------------- 落盘 --------------------

    def _write_down(self, content: str) -> Path:
        in_name = Path(self.mdf).stem
        out_path = self.fdir / f"{in_name}_filtered.md"
        out_path.write_text(content, encoding="utf-8")

        json_path = self.fdir / "figs_MetaDict.json"
        with open(json_path, "w", encoding='utf-8') as fj:
            json.dump(self.figs_info, fj, ensure_ascii=False, indent=4, default=str)

        return out_path
    

if __name__ == '__main__':
    from tqdm import tqdm 
    root_dir = r"./.log/SimplePDF"
    assert Path(root_dir).is_dir() is True
    sub_dirs = list(Path(root_dir).resolve().rglob("*/full.md"))
    # print(len(sub_dirs))  # ok 
    for md in tqdm(sub_dirs):
        mdp = Path(md)
        parsers = PatentParser(mdp)
        doc = parsers()
        
        # break 
    

