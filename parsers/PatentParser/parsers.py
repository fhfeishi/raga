# parser.py
from collections import OrderedDict
import os, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import pdfplumber



"""   
路径：专利文件路径：
    docs_patents/
        + xxxxx/   # 某一个专利的子文件夹
            + images/       # 专利markdown中的图片
            + full.md       # 专利markdown文件
            + *_origin.pdf  # 原始专利pdf文件
            + *.json  # 目前用不上的文件
        + xxxxx/   # 某一个专利的子文件夹
            + images/       # 专利markdown中的图片
            + full.md       # 专利markdown文件
            + *_origin.pdf  # 原始专利pdf文件
            + *.json  # 目前用不上的文件
        + ...
"""

class patentMD_parser:
    """
    读取专利，
    
    清理图片引用，抽取元数据，输出：
      1) MetaDict（OrderedDict，字段按给定骨架）
      2) 结构化纯文本 Markdown（<原名>_z.md）到指定目录

    结构化纯文本：
        # (54) 实用新型名称

    解析要点：
      1) 基本元数据：title / apply_time / applier / address / inventor / pubno
      2) 图片元数据：摘要图（abs_im）+ 附图（图1/图2/...）→ {"fig_list": {"图1": ["描述", "绝对路径"], ...}}
      3) 清理正文中的图片标记（![](...) + “图X ...”行）后作为向量化文本
    """

    # --------------------------- 初始化 ---------------------------
    def __init__(self, markdown_file: Union[str, Path], out_dir: Union[str, Path, None] = None):
        self.markdown_file = str(Path(markdown_file))
        self.base_dir: Path = Path(self.markdown_file).parent
        self.images_dir: Path = self.base_dir / "images"
        self.out_dir: Path = Path(out_dir) if out_dir is not None else self.base_dir
        self.pdfp = next(Path(self.markdown_file).parent.glob("*_origin.pdf"),None)
        assert Path(self.pdfp).is_file()
        self.text: str = ""  # 原始 md 文本

        self.meta_schema = OrderedDict({
            "publ_no":"",          # (10) 申请公布号 or 授权公告号   <公开号>
            "publ_date":"",        # (43) 申请公布日 or (45) 授权公告日
            "is_granted": False,    # 是否授权，授权的话才会有专利号
            "patent_no":"",        # 专利号（由申请号生成）          <专利号> if is_granted is Ture
            "apply_no": '',        # (21) 申请号(不重要)            <申请号>
            "apply_time": "",      # (22) 申请日
            "title": "",           # (54) 专利标题（实用新型/发明 名称）
            "applicant": "",      # (54) 专利权人 申请人
            "address": "",         # "邮编 地址"
            "inventors": "",       # (71) 发明人
            "doc_type": "",        # <(12) 文献类型，如“实用新型专利/发明专利申请/外观设计专利”>  # 外观设计专利只有图可以展示
            "tech_field": "",      # # 技术领域 的正文（不含标题）
            
            "root_dir": "",        # 专利目录（绝对路径）
            "pdf_path": "",        # 原始 PDF 的绝对路径
            "fig_list": {},        # {"abs_im": ["摘要图", abs_path], "图1": ["描述", abs_path], ...}
        })


    def __call__(self):
        self.pipeline()
        
    # ============== 主入口 ==============
    def pipeline(self):
        # 1) 读全文
        self.text = self._load_md_text()

        # 2) 图片元数据（摘要图 + 附图）
        img_meta = self._extract_img_metadata()  # {"fig_list": {...}}

        # 3) 把全文先裁剪到 (54) 开始；随后清理图片引用（含摘要段中的图片）
        content_from_54 = self._slice_from_54(self.text)
        text_wo_imgs = self._filter_lines(content_from_54)          # 全文去图片
        text_wo_imgs = self._clean_abstract_images(text_wo_imgs)    # 摘要段内再次兜底清图
        text_wo_imgs = self._filter_trailing_image_blocks(text_wo_imgs)  # 末尾批量图清理

        # 4) 结构化字段
        meta_blocks = self._extract_meta_blocks(self.text)  # 注意：元数据从「全文」抓，避免被裁剪掉

        # 4.1)  技术领域正文抽取（不含标题）
        tech_field_txt = self._extract_section_plain_text(self.text, r"技术领域", strip_para_tags=True)
        if tech_field_txt:
            meta_blocks["tech_field"] = tech_field_txt

        # 5) PDF 公告号 
        pubno_info = self._extract_pubno() # {"pubno": "...", "pdf_path": "..."}

        # 6) 汇总 MetaDict
        self.meta_schema["root_dir"] = str(self.base_dir.resolve())
        self.meta_schema.update(img_meta)
        self.meta_schema.update(meta_blocks)
        self.meta_schema.update(pubno_info)

        # 6-2) MetaDict_norm
        self.MetaDict_norm()


        # 7) 组织结构化 Markdown（从 (54) 起，拼装常见章节）
        structured_md = self._build_structured_markdown(text_wo_imgs, self.meta_schema)

        # # 8) 落盘, out_dir不是None的话
        if self.out_dir is not None:
            out_path = self._write_structured_md(structured_md)
            return self.meta_schema, out_path

        return self.meta_schema

    # ============== 基础工具 ==============
    def _load_md_text(self) -> str:
        return Path(self.markdown_file).read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _find_section(text: str, title_pattern: str) -> Optional[str]:
        """取指定 # 段（到下一个 # 或文末），title_pattern 不含 #。"""
        hdr = re.search(rf"^\s*#{{1,3}}\s*{title_pattern}\s*$", text, flags=re.MULTILINE)
        if not hdr:
            return None
        start = hdr.end()
        nxt = re.search(r"^\s*#\s+", text[start:], flags=re.MULTILINE)
        return text[start: start + nxt.start()] if nxt else text[start:]

    def _slice_from_54(self, text: str) -> str:
        """从 '# (54)实用新型名称'（或发明名称）开始截取；未命中则原文返回。"""
        m = re.search(r"(?m)^\s*#\s*\(54\)\s*(?:实用新型|发明)\s*名称\s*$", text)
        return text[m.start():] if m else text

    # ============== 中文数字工具（用于图号解析） ==============
    @staticmethod
    def _chs_num_to_int(s: str) -> Optional[int]:
        m = {"零":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
        s = s.strip()
        if not s: return None
        if s == "十": return 10
        if len(s) == 1 and s in m: return m[s]
        if s[0] == "十":
            tail = m.get(s[1:], 0) if s[1:] else 0
            return 10 + tail
        if "十" in s:
            left, right = s.split("十", 1)
            left_v = m.get(left, 1) if left else 1
            right_v = m.get(right, 0) if right else 0
            return left_v * 10 + right_v
        if s.isdigit(): return int(s)
        if s in m: return m[s]
        return None

    # ============== 图片元数据（摘要图 / 附图） ==============
    def _extract_abstract_img(self) -> Tuple[str, Dict[str, Dict[str, Union[List[str], None]]]]:
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

    def _parse_figure_descriptions(self) -> Dict[str, str]:
        """从『附图说明』段解析 图X→描述。"""
        desc_map: Dict[str, str] = {}
        block = self._find_section(self.text, r"附图说明")
        if not block:
            return desc_map
        block = block.replace("：", ":").replace("；", ";")
        pat = re.compile(
            r"(?:\[\d+\]\s*)?图\s*([0-9一二三四五六七八九十]+)\s*(?:为|是|:)\s*(.+?)(?=(?:；|;|。|\.|\n|$))",
            flags=re.IGNORECASE | re.DOTALL
        )
        for num, desc in pat.findall(block):
            idx = self._chs_num_to_int(num) or (int(num) if num.isdigit() else None)
            if not idx:
                continue
            key = f"图{idx}"
            desc_map[key] = re.sub(r"[\s;；。.\u3000]+$", "", desc.strip())
        return desc_map

    def _build_fig_map(self) -> Dict[str, str]:
        """匹配形式：图片行 →（可有空行）→ 下一行 '图N'。"""
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

    def _extract_img_metadata(self) -> Dict[str, Dict[str, Union[List[str], None]]]:
        """整合摘要图 + 附图说明 + 图片路径为 fig_list。"""
        _, abs_meta = self._extract_abstract_img()
        abs_item = abs_meta["fig_list"].get("abs_im")
        desc_map = self._parse_figure_descriptions()
        img_map  = self._build_fig_map()
        fig_list: Dict[str, Union[List[str], None]] = {}
        for k, path in img_map.items():
            desc = desc_map.get(k, "")
            fig_list[k] = [desc, path]
        if abs_item and isinstance(abs_item, list) and len(abs_item) == 2:
            fig_list["abs_im"] = abs_item
        return {"fig_list": fig_list}

    # ============== 文本清理 ==============
    def _filter_lines(self, md_text: str) -> str:
        """
        清除所有图片引用 + 紧随的“图X …”行；再清除**孤立的“图N”整行**。
        """
        # 1) 图片 + 紧随“图N …”的一行
        pattern = re.compile(
            r'!\[.*?\]\([^)]*\)[ \t]*(?:\n[ \t]*)*\n?[ \t]*图[0-9一二三四五六七八九十]+[：:\s]*[^\n]*(?:\n|$)',
            flags=re.IGNORECASE | re.MULTILINE
        )
        out = pattern.sub('', md_text)
        # 2) 裸图片
        out = re.sub(r'!\[.*?\]\([^)]*\)\s*\n?', '', out)
        # 3) NEW: 孤立“图N”整行（只保留纯正文里诸如“如图2所示”，不会删掉这类）
        out = re.sub(r'(?m)^\s*图[0-9一二三四五六七八九十]+\s*$', '', out)
        out = re.sub(r'(?m)^\s*图[0-9一二三四五六七八九十]+\s*[：:]\s*$', '', out)
        # 4) 清理多余空行（可选）
        out = re.sub(r'\n{3,}', '\n\n', out).strip() + "\n"
        return out

    def _clean_abstract_images(self, md_text: str) -> str:
        """专门对 (57)摘要 段做兜底去图（防止摘要内还有残留图片标记）。"""
        sec = self._find_section(md_text, r"\(?57\)?\s*摘要")
        if not sec:
            return md_text
        sec_clean = re.sub(r'!\[.*?\]\([^)]*\)', '', sec).strip()
        return md_text.replace(sec, sec_clean)

    def _filter_trailing_image_blocks(self, text: str) -> str:
        """
        删除文末连续的图块。
        支持两种形式反复出现：
          a) [图片] (+空行) + 图N...
          b) 纯“图N...”行（没有配图）
        """
        new_text = re.sub(
            r'('
            r'\n\s*(?:!\[.*?\]\([^)]+\)\s*)?(?:\n[ \t]*)*图[0-9一二三四五六七八九十]+[^\n]*\s*'
            r')+$',
            '',
            text,
            flags=re.MULTILINE | re.IGNORECASE
        ).rstrip() + "\n"
        return new_text
    
    # ============== 时间格式 ==============
    def _normalize_date(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        s = s.replace("年", ".").replace("月", ".").replace("日", "")
        s = s.replace("/", ".").replace("-", ".")
        s = re.sub(r'\.+', '.', s).strip(".")
        return s

    # ============== 元数据抽取（从『全文』提取，避免被裁剪掉） ==============
    def _extract_meta_blocks(self, full_text: str) -> Dict[str, str]:
        
        # (12) 文献类型
        m12 = re.search(r'\(12\)\s*([^\n]+)', full_text)
        if m12:
            doc_type = re.sub(r"\s+", "", m12.group(1))
            
        
        # (54) 标题：下一行是 title
        m_title = re.search(r'(?m)^#\s*\(54\)\s*(?:实用新型|发明)\s*名称\s*\n(.+)$', full_text)
        title = (m_title.group(1).strip() if m_title else "")

        # (21) 申请号
        m_apply_no = re.search(r'\(21\)\s*申请号\s*([0-9]+(?:\.[0-9A-Za-z])?)', full_text)
        apply_no = (m_apply_no.group(1).strip() if m_apply_no else "")

        # (22) 申请日：形如 2020.09.02 或 2020-09-02 或 2020年09月02日
        m_apply = re.search(r'\(22\)\s*申请日\s*([0-9.\-年月日/]+)', full_text)
        apply_time = (m_apply.group(1).strip() if m_apply else "")
        apply_time = apply_time.replace("年",".").replace("月",".").replace("日","").strip(".")

        # (73) 专利权人  or (71)申请人   +   地址（可能连写：地址<邮编><地址>）   gg  
        applicant, address = "", ""
        # 统一从 (73)/(71) 段落中抽取，允许跨行直到下一个 (XX) 字段或文末
        block_pat = re.compile(
                r'\((?P<code>71|73)\)'                    # (71) 或 (73)
                r'\s*(?:申\s*请\s*人|专\s*利\s*权\s*人)?'  # 可选“申请人/专利权人”字样
                r'\s*[:：]?\s*'
                r'(?P<name>.*?)'                          # 申请人/专利权人（非贪婪）
                r'(?:地址|住\s*址)\s*[:：]?\s*'            # 地址提示词
                r'(?:(?P<zip>\d{6})\s*[，,、]?\s*)?'      # 可选 6 位邮编
                r'(?P<addr>[^\(\)\r\n]+)',                # 地址主体
                flags=re.S | re.I
            )
        m = block_pat.search(full_text)
        if m:
            applicant = m.group('name').strip()
            zip_code = m.group('zip') or ''
            addr = m.group('addr').strip()
            address = f"{zip_code} {addr}".strip()
        
        # # 2. 兜底：单行抓“申请人/专利权人”
        # for tag in ('(71)', '(73)'):
        #     m_name = re.search(rf'{re.escape(tag)}\s*(?:申\s*请\s*人|专\s*利\s*权\s*人)?\s*[:：]?\s*([^\r\n]+)', full_text, re.I)
        #     if m_name:
        #         applicant = m_name.group(1).strip()
        #         break
        
        # # 3. 兜底：单行抓“地址”
        # m_addr = re.search(r'(?:地址|住\s*址)\s*[:：]?\s*(\d{6})?[，,、]?\s*([^\r\n]+)', full_text, re.I)
        # if m_addr:
        #     zip_code = m_addr.group(1) or ''
        #     addr = m_addr.group(2).strip()
        #     address = f"{zip_code} {addr}".strip()
              

        # (72) 发明人 or 设计人 → 多个人名，返回 ['xx', 'xx', ...]   --gg  解析还是有点问题
        inventors_list = []
        m_72 = re.search(
            # 抓住 (72) 行后直到“下一字段头（形如 \n(XX)）”或文末为止
            r'(?ms)^\(72\)\s*(?:发明人|设计人)\s*(.+?)(?=\n\(\d{2}\)|\Z)',
            full_text
        )
        if m_72:
            raw_inv = m_72.group(1)

            # 统一空白（包含全角空格/换行等）为单空格
            raw_inv = re.sub(r'\s+', ' ', raw_inv).strip()

            # 先把常见分隔符与连接词统一成 "、"
            # 顿号/逗号/分号/斜杠 以及 “和/与/及” 两侧可能有空格
            normalized = re.sub(r'\s*(?:、|，|,|；|;|／|/|和|与|及)\s*', '、', raw_inv)

            # 关键：中文名之间仅以空格分隔的情况，也将这个空格视为一个分隔符
            normalized = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '、', normalized)

            # 分割并清洗
            parts = [p.strip() for p in normalized.split('、') if p.strip()]

            seen = set()
            for p in parts:
                # 去掉尾缀“等/等人/等等人”等
                p = re.sub(r'(?:等(?:等)?(?:人)?)$', '', p)
                if p and p not in seen:
                    inventors_list.append(p)
                    seen.add(p)

        # 结果：列表形式
        inventors = inventors_list

        #  (43)/(45)/关键词“公开/公告/公布/授权公告(日|日期)” —— 取最先匹配到的
        pub_date = ""
        for pat in [
            r'\(45\)\s*授权公告日\s*([0-9.\-年月日/]+)',
            r'\(43\)\s*(?:公开|公告|公布)\s*(?:日|日期)?\s*([0-9.\-年月日/]+)',
            r'(?:授权公告|公开|公告|公布)\s*(?:日|日期)\s*[:：]?\s*([0-9.\-年月日/]+)',
        ]:
            m = re.search(pat, full_text)
            if m:
                pub_date = self._normalize_date(m.group(1))
                break
        



        return {
            "title": title,
            "apply_no":apply_no,
            "apply_time": apply_time,
            "applicant": applicant,
            "address": address,
            "inventors": inventors,
            "publ_date": pub_date,  
            "doc_type":doc_type, 
        }

    def _extract_pub_date_from_pdf(self) -> str:
        """  
        1) 优先匹配 (45)授权公告日 / 公开|公告|公布(日|日期)
        2)从 PDF 首页兜底提取 pub_date：

        
        
        
        """
        
    def _extract_pubno_from_pdf(self) -> str:
        pass 
        
    
    # <公布号/公告号>  ->  授权 专利号
    def MetaDict_norm(self) -> None:
        publ_no = self.meta_schema.get("publ_no", None)
        if not publ_no: 
            # 打印版的PDF文件，没法
            # 既然不知道 <公布号/公告号>， 也无法得到 patent_no
            del self.meta_schema['patent_no']
            return 
        
        appl_no = self.meta_schema.get("apply_no", None)
        if not appl_no: 
            return 
        pub_kind = publ_no.strip()[-1]  # 取<公开号>最后一位
        # A  发明专利申请，未授权， 无专利号
        # U/C/S 实用新型/发明/外观设计授权, 授权， 有专利号
        if pub_kind in ['U', 'C', 'S']:
            patent_no = f"ZL{appl_no}"  # 授权，专利号=ZL申请号
            self.meta_schema['patent_no'] = patent_no
            self.meta_schema['is_granted'] = True 
        elif pub_kind == 'A':
            del self.meta_schema['patent_no']
        else:
            del self.meta_schema['patent_no']
        return 

    # 提取指定标题段的“纯正文”（不含标题本身）
    def _extract_section_plain_text(self, full_text: str, title_pattern: str, strip_para_tags: bool = False) -> str:
        sec = self._find_section(full_text, title_pattern)
        if not sec:
            return ""
        txt = sec.strip()
        if strip_para_tags:
            # 去掉每段行首的 [0001]、[0017] 等
            txt = re.sub(r'(?m)^\s*\[\d+\]\s*', '', txt)
        return txt

    # ============== PDF 公告号 / PDF 路径 ==============
    def _guess_pdf_path(self) -> Optional[str]:
        cands = list(self.base_dir.glob("*_origin.pdf"))
        if not cands:
            cands = list(self.base_dir.glob("*.pdf"))
        return str(cands[0]) if cands else None

    def _extract_pubno(self) -> Dict[str, str]:
        """
        直接从 markdown 文本中提取公告号/授权公告号（如 CN110405743A、CN211234567U、CN123456789B1）。
        提取顺序：
        1) (11) 行（公开号/公告号/授权公告号）
        2) 带“公告号/公开号/授权公告号”关键词的行
        """
        text = self.text

        # 1) (11) 行优先
        m_11 = re.search(r'\(11\)[^\n]*?(CN[0-9]{7,12}[A-Z0-9]?)', text, flags=re.IGNORECASE)
        if m_11:
            return {"publ_no": m_11.group(1).upper(), "pdf_path": self._guess_pdf_path() or ""}

        # 2) 关键词行（公告号/公开号/授权公告号）
        m_kw = re.search(
            r'(公告号|公布号|申请公布号|授权公告号)\s*[:：]?\s*(CN[0-9]{7,12}[A-Z0-9]?)',
            text, flags=re.IGNORECASE
        )
        if m_kw:
            return {"publ_no": m_kw.group(2).upper(), "pdf_path": self._guess_pdf_path() or ""}

        # 都没找到则置空
        return {"publ_no": "", "pdf_path": self._guess_pdf_path() or ""}
    
    
    
    
    
    # ============== 结构化 Markdown 生成（从 (54) 起） ==============
    def _build_structured_markdown(self, clean_text_from_54: str, meta: Dict[str, Union[str, dict]]) -> str:
        """
        clean_text_from_54：已从 (54) 起切片 + 去图 + 末尾图清理
        章节顺序（存在即纳入）：
          # (54) 实用新型名称
          # (57) 摘要   （若摘要段内出现条款编号，从此处切出到“权力说明书”）
          # 权力说明书 （NEW：当摘要后直接出现 1./一、 等编号条款，且无“权利要求”标题时）
          # 技术领域 / 背景技术 / 实用新型内容
          # 权利要求（若原文有“权利要求书/权利要求”标题）
          # 说明书 / 具体实施方式 / 发明内容
          # 附图说明（文字描述）
        """
        lines: List[str] = []

        # (54) 名称
        title = str(meta.get("title") or "").strip()
        if not title:
            m_title = re.search(r'(?m)^#\s*\(54\)\s*(?:实用新型|发明)\s*名称\s*\n(.+)$', clean_text_from_54)
            title = (m_title.group(1).strip() if m_title else "")
        if title:
            lines += ["# (54) 实用新型名称", title, ""]

        # (57) 摘要（从 clean_text_from_54 内重找一次，确保无图片）
        abs_block = self._find_section(clean_text_from_54, r"\(?57\)?\s*摘要")
        claims_from_abs = ""  # NEW: 从摘要中切出来的条款
        if abs_block:
            # NEW: 识别条款起始（1./1．/1、/一、/十一、 等）
            m_claim_start = re.search(r'(?m)^\s*(?:\d+|[一二三四五六七八九十]+)[\.\．、]\s', abs_block)
            if m_claim_start:
                abs_text = abs_block[:m_claim_start.start()].strip()
                claims_from_abs = abs_block[m_claim_start.start():].strip()
            else:
                abs_text = abs_block.strip()
            lines += ["# (57) 摘要", abs_text, ""]
        else:
            lines += ["# (57) 摘要", ""]

        # 权利要求（显式标题）
        claims = self._find_section(clean_text_from_54, r"(权利要求书|权利要求)")

        if claims:
            lines += ["# 权利要求", claims.strip(), ""]
        elif claims_from_abs:
            # NEW: 没有“权利要求”标题，但摘要后直接给了条款，则落到“权力要求书”
            lines += ["# 权力要求书", claims_from_abs, ""]

        # 技术领域 / 背景技术 / 实用新型内容（逐个纳入）
        for pat, nice in [
            (r"技术领域", "技术领域"),
            (r"背景技术", "背景技术"),
            (r"实用新型内容", "实用新型内容"),
        ]:
            sec = self._find_section(clean_text_from_54, pat)
            if sec and sec.strip():
                lines += [f"# {nice}", sec.strip(), ""]

        # 说明书系列（择优加入，避免重复）
        seen = set()
        for pat, title_ in [
            (r"说明书", "说明书"),
            (r"具体实施方式", "具体实施方式"),
            (r"发明内容", "发明内容"),
        ]:
            sec = self._find_section(clean_text_from_54, pat)
            if sec and sec.strip() and title_ not in seen:
                lines += [f"# {title_}", sec.strip(), ""]
                seen.add(title_)

        # 附图说明（仅文字描述，来自解析）
        fig_desc_map = self._parse_figure_descriptions()
        if fig_desc_map:
            lines += ["# 附图说明"]
            for k in sorted(fig_desc_map, key=lambda x: int(re.sub(r"\D", "", x) or "0")):
                lines.append(f"{k}：{fig_desc_map[k]}")
            lines.append("")

        text_final = "\n".join(lines).rstrip() + "\n"
        return text_final

    # ============== 写文件 ==============
    def _write_structured_md(self, content: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        in_name = Path(self.markdown_file).stem
        out_path = self.out_dir / f"{in_name}_z.md"
        out_path.write_text(content, encoding="utf-8")
        return out_path
        


if __name__ == '__main__':
    # markdown_file = r"./demo.md"
    # parser = patentMD_parser(markdown_file)
    # doc = parser()
    # print(parser.meta_schema)
    
    root_dir = r"D:\ddesktop\agentdemos\codespace\zhuanliParser\result"
    od = r"D:\ddesktop\agentdemos\codespace\zhuanliParser\result_1"
    sub_dirs = list(Path(root_dir).glob("*/full.md"))
    print(len(sub_dirs))  # ok 
    for md in sub_dirs:
        mdp = Path(md)
        pdfp = next(mdp.parent.glob("*_origin.pdf"),None)
        assert pdfp is not None 
        parsers = patentMD_parser(mdp, od)
        doc = parsers()
        print(parsers.meta_schema)
    
    
    
    

