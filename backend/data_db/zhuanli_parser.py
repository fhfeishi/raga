from collections import OrderedDict
import os, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

from pypdf import PdfReader
from langchain_core.documents import Document


class zhuanli_parser:
    """
    将专利解析后的 full.md 转为 LangChain Document，并抽取结构化元数据。
    目录结构（示例）：
        <patent_dir>/
          ├─ full.md
          ├─ xx_origin.pdf            # 原始专利PDF（推荐命名）
          └─ images/
              ├─ a.jpg
              ├─ b.jpg
              └─ ...

    解析要点：
      1) 基本元数据：title / apply_time / applier / address / inventor / pubno
      2) 图片元数据：摘要图（abs_im）+ 附图（图1/图2/...）→ {"fig_list": {"图1": ["描述", "绝对路径"], ...}}
      3) 清理正文中的图片标记（![](...) + “图X ...”行）后作为向量化文本
    """

    # --------------------------- 初始化 ---------------------------
    def __init__(self, markdown_file: str):
        self.markdown_file = str(Path(markdown_file))
        self.base_dir: Path = Path(self.markdown_file).parent
        self.images_dir: Path = self.base_dir / "images"

        # 解析过程中的缓存文本（pipeline 内赋值）
        self.text: str = ""

        # 目标元数据骨架（有序，方便可视化/调试）
        self.meta_schema = OrderedDict({
            "pubno": "",           # 授权/公告号，如 CN20xxxx
            "title": "",           # 专利标题（(54) 实用新型/发明 名称）
            "applier": "",         # 专利权人（只保留公司名）
            "address": "",         # 地址字段单独保存
            "inventor": "",        # 发明人
            "apply_time": "",      # (22) 申请日
            "root_dir": str(self.base_dir.resolve()),  # 专利目录
            "pdf_path": "",        # 原始 PDF 的绝对路径（尽力猜测）
            "fig_list": {},        # {"abs_im": ["摘要图", abs_path], "图1": ["描述", abs_path], ...}
        })

    # --------------------------- 主入口 ---------------------------
    def __call__(self) -> Document:
        return self.pipeline()

    # --------------------------- 工具：加载全文 ---------------------------
    def _load_md_text(self) -> str:
        return Path(self.markdown_file).read_text(encoding="utf-8")

    # --------------------------- 工具：中文数字 → 阿拉伯数字 ---------------------------
    @staticmethod
    def _chs_num_to_int(s: str) -> Optional[int]:
        """
        仅处理 1~99 的常见中文数字（十、十一、二十、二十三…），够用即可。
        """
        m = {"零":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
        s = s.strip()
        if not s: return None
        if s == "十": return 10
        if len(s) == 1 and s in m: return m[s]
        # 十X
        if s[0] == "十":
            tail = m.get(s[1:], 0) if s[1:] else 0
            return 10 + tail
        # X十 / X十Y
        if "十" in s:
            left, right = s.split("十", 1)
            left_v = m.get(left, 1) if left else 1
            right_v = m.get(right, 0) if right else 0
            return left_v * 10 + right_v
        if s.isdigit(): return int(s)
        if s in m: return m[s]
        return None

    # --------------------------- 工具：截取 # 标题段 ---------------------------
    @staticmethod
    def _find_section(text: str, title_pattern: str) -> Optional[str]:
        r"""
        获取形如 "# 附图说明" / "# (57) 摘要" 段落（到下一个 # 或文末）。
        title_pattern：不含 # 的正则，如 r'附图说明' 或 r'\(57\)\s*摘要'
        """
        hdr = re.search(rf"^\s*#{{1,3}}\s*{title_pattern}\s*$", text, flags=re.MULTILINE)
        if not hdr:
            return None
        start = hdr.end()
        nxt = re.search(r"^\s*#\s+", text[start:], flags=re.MULTILINE)
        return text[start: start + nxt.start()] if nxt else text[start:]

    # --------------------------- 图片：摘要图 ---------------------------
    def _extract_abstract_img(self) -> Tuple[str, Dict[str, List[str]]]:
        """
        返回: (cleaned_abstract_text, {"fig_list": {"abs_im": ["摘要图", abs_path]}})
        若无摘要图，abs_im 为 None
        """
        block = self._find_section(self.text, r"\(?57\)?\s*摘要")
        if not block:
            return "", {"fig_list": {"abs_im": None}}

        img_m = re.search(r'!\[.*?\]\((.*?)\)', block)
        if not img_m:
            return block.strip(), {"fig_list": {"abs_im": None}}

        rel_path = img_m.group(1).strip()
        abs_path = str((self.base_dir / rel_path).resolve())
        cleaned = re.sub(r'!\[.*?\]\(.*?\)\s*\n?', "", block).strip()
        return cleaned, {"fig_list": {"abs_im": ["摘要图", abs_path]}}

    # --------------------------- 图片：附图说明（图号→描述） ---------------------------
    def _parse_figure_descriptions(self) -> Dict[str, str]:
        """
        返回 {'图1': 'xxx', '图2': 'yyy', ...}
        兼容：
          - "[0017] 图1为……；[0018] 图2：……。"
          - "图一是……；图二为……。"
          - "图1: ……" / "图1：……"
        """
        desc_map: Dict[str, str] = {}
        block = self._find_section(self.text, r"附图说明")
        if not block:
            return desc_map

        # 统一标点，便于匹配
        block = block.replace("：", ":").replace("；", ";")

        pat = re.compile(
            r"(?:\[\d+\]\s*)?"                # 可选编号 [0017]
            r"图\s*([0-9一二三四五六七八九十]+)\s*"
            r"(?:为|是|:)\s*"
            r"(.+?)"                          # 描述
            r"(?=(?:；|;|。|\.|\n|$))",        # 句末/换行/文末
            flags=re.IGNORECASE | re.DOTALL
        )

        for num, desc in pat.findall(block):
            idx = self._chs_num_to_int(num) or (int(num) if num.isdigit() else None)
            if not idx:
                continue
            key = f"图{idx}"
            desc_map[key] = re.sub(r"[\s;；。.\u3000]+$", "", desc.strip())

        return desc_map

    # --------------------------- 图片：全文图片（图号→路径） ---------------------------
    def _build_fig_map(self) -> Dict[str, str]:
        """
        全文扫描图片，尝试关联到后续最近的“图X”标签。
        兼容：图片与“图X”在同一行 / 下一行 / 连续两行软换行。
        返回 {'图1': abs_path, ...}
        """
        img_map: Dict[str, str] = {}

        img_iter = re.finditer(
            r'!\[.*?\]\((.*?)\)\s*(?:\n[ \t]*){0,2}(图[0-9一二三四五六七八九十]+)',
            self.text, flags=re.IGNORECASE
        )
        for m in img_iter:
            rel_path = m.group(1).strip()
            tag = m.group(2).strip()
            mnum = re.match(r"图\s*([0-9一二三四五六七八九十]+)", tag)
            if not mnum:
                continue
            idx_raw = mnum.group(1)
            idx = self._chs_num_to_int(idx_raw) or (int(idx_raw) if idx_raw.isdigit() else None)
            if not idx:
                continue
            key = f"图{idx}"
            abs_path = str((self.base_dir / rel_path).resolve())
            if Path(abs_path).exists():
                img_map[key] = abs_path

        return img_map

    # --------------------------- 图片：整合 摘要图 + 附图说明 + 路径 ---------------------------
    def _extract_img_metadata(self) -> Dict[str, Dict[str, List[str]]]:
        """
        返回:
          {"fig_list": {
              "abs_im": ["摘要图", <abs_path>] 或 None,
              "图1": ["描述或空字符串", <abs_path>], ...
          }}
        """
        _, abs_meta = self._extract_abstract_img()   # {"fig_list": {"abs_im": [...]} 或 None}
        abs_item = abs_meta["fig_list"].get("abs_im")

        desc_map = self._parse_figure_descriptions() # {'图1': '...', ...}
        img_map  = self._build_fig_map()             # {'图1': 'abs_path', ...}

        fig_list: Dict[str, List[str]] = {}
        for k, path in img_map.items():
            desc = desc_map.get(k, "")
            fig_list[k] = [desc, path]

        if abs_item and isinstance(abs_item, list) and len(abs_item) == 2:
            fig_list["abs_im"] = abs_item

        return {"fig_list": fig_list}

    # --------------------------- 文本：清理图片标记 ---------------------------
    def _filter_lines(self, md_text: str) -> str:
        """
        删除形如：
           ![...](...)
           图X...
        的成对块（支持中间空行）。摘要图在向量化阶段通常也不需要，可一并清除。
        """
        pattern = re.compile(
            r'!\[.*?\]\([^)]*\)[ \t]*(?:\n[ \t]*)*\n?[ \t]*图[0-9一二三四五六七八九十]+[：:\s]*[^\n]*(?:\n|$)',
            flags=re.IGNORECASE | re.MULTILINE
        )
        out = pattern.sub('', md_text)
        # 同时清理裸图片（摘要图等未伴随“图X”标注）
        out = re.sub(r'!\[.*?\]\([^)]*\)\s*\n?', '', out)
        return out.strip() + "\n"

    # --------------------------- 元数据：PDF 公告号提取 ---------------------------
    def _extract_pubno(self) -> Dict[str, str]:
        """
        从原始 PDF 首页末行提取公告号（CN...）。若目录下存在 *_origin.pdf 优先取之；
        否则尝试取当前目录下任一 .pdf；找不到则抛异常提示。
        """
        pdf_path = self._guess_pdf_path()
        if not pdf_path:
            raise FileNotFoundError(f"未找到原始 PDF 文件（建议命名 *_origin.pdf），目录：{self.base_dir}")

        reader = PdfReader(pdf_path)
        text_1 = reader.pages[0].extract_text() or ""
        last_line = (text_1.strip().splitlines() or [""])[-1]

        compact = re.sub(r'\s+', '', last_line.upper())
        m = re.search(r'(CN[A-Z0-9]{9,13})', compact)
        if m:
            return {"pubno": m.group(1), "pdf_path": str(Path(pdf_path).resolve())}
        else:
            # 个别版式不在页尾，则退化：全页扫描一次
            compact_all = re.sub(r'\s+', '', text_1.upper())
            m2 = re.search(r'(CN[A-Z0-9]{9,13})', compact_all)
            if m2:
                return {"pubno": m2.group(1), "pdf_path": str(Path(pdf_path).resolve())}
            raise ValueError(f"PDF 未找到公告号 CN**** ：{os.path.basename(pdf_path)}")

    def _guess_pdf_path(self) -> Optional[str]:
        """
        猜测原始 PDF 路径：
          1) 同目录 *_origin.pdf
          2) 同目录 *.pdf
        """
        cands = list(self.base_dir.glob("*_origin.pdf"))
        if not cands:
            cands = list(self.base_dir.glob("*.pdf"))
        return str(cands[0]) if cands else None

    # --------------------------- 元数据：(54)/(22)/(73)/(72) ---------------------------
    def _extract_meta_blocks(self, text: str) -> Dict[str, str]:
        """
        提取：
          - title: (54) 实用新型名称 / 发明名称
          - apply_time: (22) 申请日
          - applier, address: (73) 专利权人（名称、地址分离）
          - inventor: (72) 发明人
        针对 OCR/解析误差做了尽量鲁棒的正则。
        """
        # (54) 标题（“实用新型名称 / 发明名称”均兼容）
        m_title = re.search(r'(?m)^#\s*\(54\)\s*(?:实用新型|发明)名称\s*\n(.+)$', text)
        title = m_title.group(1).strip() if m_title else ""

        # (22) 申请日
        m_apply = re.search(r'\(22\)\s*申请日\s*([0-9.\-年月日/]+)', text)
        apply_time = (m_apply.group(1).strip() if m_apply else "").replace("年","." ).replace("月",".").replace("日","").strip(".")

        # (73) 专利权人：尽量分离“地址”
        # 典型："(73)专利权人 杭州宇树科技有限公司 地址 310051浙江省杭州市..."
        m_73 = re.search(r'\(73\)\s*专利权人\s*([^\n]+)', text)
        applier = ""
        address = ""
        if m_73:
            line = re.sub(r'\s+', '', m_73.group(1))
            # 优先按“地址”切分
            if "地址" in line:
                parts = line.split("地址", 1)
                applier = parts[0].strip()
                address = parts[1].strip()
            else:
                # 若无“地址”，退化：取到空白前的公司名（常见公司后缀）
                m_company = re.match(r'(.+?(?:公司|研究院|大学|学院|研究所|中心))', line)
                applier = m_company.group(1) if m_company else line

        # (72) 发明人
        m_72 = re.search(r'\(72\)\s*发明人\s*([^\n]+)', text)
        inventor = (m_72.group(1).strip() if m_72 else "")

        return {
            "title": title,
            "apply_time": apply_time,
            "applier": applier,
            "address": address,
            "inventor": inventor
        }

    # --------------------------- 文末图片块清理（可选） ---------------------------
    def _filter_trailing_image_blocks(self, text: str) -> str:
        """
        删除末尾连续出现的 “图片 + 图X” 块（若你的 full.md 末尾集中放图，可开启此步）。
        """
        new_text = re.sub(
            r'(\n\s*!\[.*?\]\([^)]+\)\s*\n\s*图[0-9一二三四五六七八九十]+\s*)+$',
            '',
            text,
            flags=re.MULTILINE | re.IGNORECASE
        ).rstrip() + '\n'
        return new_text

    # --------------------------- 总流水线 ---------------------------
    def pipeline(self) -> Document:
        # 1) 读取全文
        self.text = self._load_md_text()

        # 2) 图片元数据（摘要图 + 附图）
        img_meta = self._extract_img_metadata()        # {"fig_list": {...}}

        # 3) 清理正文图片标记，得到纯文本（供向量化）
        md_text_filtered = self._filter_lines(self.text)

        # 4) 结构化字段：title/apply_time/applier/address/inventor
        meta_blocks = self._extract_meta_blocks(md_text_filtered)

        # 5) PDF 公告号 & pdf_path
        pubno_info = self._extract_pubno()             # {"pubno": "...", "pdf_path": "..."}

        # 6) 汇总元数据（保持有序）
        self.meta_schema.update(img_meta)
        self.meta_schema.update(meta_blocks)
        self.meta_schema.update(pubno_info)

        # 7) 组装为 LangChain Document
        doc = Document(page_content=md_text_filtered, metadata=dict(self.meta_schema))
        return doc
        


if __name__ == '__main__':
    markdown_file = r"D:\ddesktop\agentdemos\codespace\zhuanliParser\result\CN202021894937.5-一种结构紧凑的回转动力单元以及应用其的机器人.pdf-c96c9eb4-261f-46aa-8b93-8211ff1d937d\full.md"
    parser = zhuanli_parser(markdown_file)
    doc = parser()
    # 打印元数据（含图像字典）
    for k, v in doc.metadata.items():
        print(f"{k}: {v}")

