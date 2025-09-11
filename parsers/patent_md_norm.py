# -*- coding: utf-8 -*-
"""
将“专利文本（粗糙 Markdown）”规范化为结构化 Markdown：
- 归并“著录信息”（含摘要）
- 补齐“权利要求书”一级标题
- 归档“说明书”并抽取子段（标题、背景技术、发明/实用新型内容、具体实施方式）
- 忽略“附图说明 / 说明书附图”段（按用户要求）
"""


from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CLAIM_LINE_RE = re.compile(r"^\s*\d+[\.\．、\)]\s*")  # 1. / 1、 / 1) / 1．
ABS_RE = re.compile(r"\(57\)\s*摘要")  # 摘要锚点（可包含井号）
TYPE_UTILITY_RE = re.compile(r"\(12\).*?实用新型")
TYPE_INVENTION_RE = re.compile(r"\(12\).*?发明")
NAME_54_RE = re.compile(r"\(54\).{0,6}(名称|题名)")
HEADING_HASH_RE = re.compile(r"^\s*#+\s*(.+?)\s*$")


@dataclass
class PatentPieces:
    patent_type: str     # 'utility' | 'invention' | 'unknown'   unknown = "外观设计专利"
    meta_lines: List[str]
    abstract_lines: List[str]
    claims_lines: List[str]
    body_lines: List[str]
    body_sections: Dict[str, List[str]]
    title_text: Optional[str]


class PatentMDNorm:
    """
    用于将 PDF 提取的“粗糙专利 Markdown”规范化为结构化 Markdown。
    仅面向“发明专利 / 实用新型专利”的常见格式。
    
    # 著录信息
    ## (xx)..:
    ...
    ## 
    ... 
    
    # 权力要求书
    1. ...
    2. ...
    ...    
    
    # 说明书
    ## 标题
    ...
    ## 背景技术
    ...
    ## 发明内容/实用新型内容
    ...
    ## 具体实施方式    <--link--> 图片figs_MetaDict.json 
    ...
    
    """
    def __init__(self, raw:str, out_path: str = "zz.md"):
        # mdp: Path      out_suffix="_norm.md"
        # self.raw_md = Path(mdp).read_text(encoding='utf-8', errors='ignore').replace("\r\n", "\n")
        # self.out_path = Path(mdp).parent / Path(mdp).stem / out_suffix
        self.raw_md = raw.replace("\r\n", "\n")
        self.out_path = Path(out_path)

    # -------------------------- 公共 API --------------------------
    def to_markdown(self) -> str:
        p = self._segment(self.raw_md)
        return self._compose_markdown(p)

    def write(self, path: Optional[str] = None) -> Path:
        md = self.to_markdown()
        path = Path(path) if path else self.out_path
        path.write_text(md, encoding="utf-8")
        return path

    # -------------------------- 内部逻辑 --------------------------
    def _segment(self, text: str) -> PatentPieces:
        lines = text.split("\n")

        # 1) 类型判定
        patent_type = "unknown"
        if any(TYPE_UTILITY_RE.search(l) for l in lines[:60]):
            patent_type = "utility"
        elif any(TYPE_INVENTION_RE.search(l) for l in lines[:60]):
            patent_type = "invention"

        # 2) 找“(57)摘要”锚点
        abs_idx = None
        for i, ln in enumerate(lines):
            if ABS_RE.search(ln):
                abs_idx = i
                break

        # meta（含“摘要段”之前的所有著录信息 + “摘要正文”）
        meta_lines: List[str] = []
        abstract_lines: List[str] = []

        # claims 与 body 预留
        claims_lines: List[str] = []
        body_lines: List[str] = []

        if abs_idx is None:
            # 没有 (57) 摘要锚点：保守落地——全部先当 meta，后续尽力分段
            meta_lines = lines
        else:
            # meta 为开头至摘要标题行
            meta_lines = lines[:abs_idx + 1]
            # 摘要正文：从下一行开始，直到首个权利要求条款或“说明书”类标题
            j = abs_idx + 1
            while j < len(lines) and not CLAIM_LINE_RE.match(lines[j]) \
                    and not self._is_body_heading(lines[j]):
                abstract_lines.append(lines[j])
                j += 1

            # 3) 权利要求书：从首个编号条款开始，直到遇到说明书类标题
            k = j
            while k < len(lines) and CLAIM_LINE_RE.match(lines[k]):
                claims_lines.append(lines[k])
                k += 1

            # 4) 余下即“说明书”主体
            body_lines = lines[k:]

        # 5) 抽取说明书子段
        body_sections = self._extract_body_sections(body_lines)

        # 6) 提取标题文本（优先 (54) 名称 -> 次选“居中标题/独立一级标题”）
        title_text = self._extract_title_text(lines, body_sections)

        return PatentPieces(
            patent_type=patent_type,
            meta_lines=meta_lines,
            abstract_lines=self._trim_blank_edges(abstract_lines),
            claims_lines=self._trim_blank_edges(claims_lines),
            body_lines=body_lines,
            body_sections=body_sections,
            title_text=title_text,
        )

    def _compose_markdown(self, p: PatentPieces) -> str:
        # 规范化“著录信息”区：将开头的 # (xx) 或纯文本行，统一转为二级小节
        meta_block = self._format_meta_block(p.meta_lines, p.abstract_lines)

        # “权利要求书”区：若未检出条款，保留空壳并给出中文占位提醒
        claims_block = ["# 权利要求书"]
        if p.claims_lines:
            claims_block.extend(p.claims_lines)
        else:
            claims_block.append("（未在源文档中稳定识别到条款编号；请人工核对后补齐——此处为占位提示，需替换。）")

        # “说明书”区：收拢必须子段，并按类型选择“发明内容/实用新型内容”
        body_block = ["# 说明书"]

        # 标题
        title_text = p.title_text or "（此处需替换：未能可靠提取标题）"
        body_block.append("## 标题")
        body_block.append(title_text)

        # 技术领域
        if p.body_sections.get("技术领域"):
            body_block.append("## 技术领域")
            body_block.extend(p.body_sections["技术领域"])

        # 背景技术
        if p.body_sections.get("背景技术"):
            body_block.append("## 背景技术")
            body_block.extend(p.body_sections["背景技术"])

        # 发明内容 / 实用新型内容
        content_key = "发明内容" if (p.patent_type == "invention") else "实用新型内容"
        # 兜底：若没有匹配，尝试两者之一
        alt_key = "实用新型内容" if content_key == "发明内容" else "发明内容"
        content_lines = p.body_sections.get(content_key) or p.body_sections.get(alt_key)
        if content_lines:
            body_block.append(f"## {content_key if content_lines == p.body_sections.get(content_key) else alt_key}")
            body_block.extend(content_lines)

        # 具体实施方式（可多个块，合并）
        impl = p.body_sections.get("具体实施方式")
        if impl:
            body_block.append("## 具体实施方式")
            body_block.extend(impl)

        # 组合输出（忽略“附图说明/说明书附图”）
        parts = []
        parts.extend(meta_block)
        parts.append("")

        parts.extend(claims_block)
        parts.append("")

        parts.extend(body_block)
        parts.append("")

        return "\n".join(parts)

    # -------------------------- 工具函数 --------------------------
    def _format_meta_block(self, meta_lines: List[str], abstract_lines: List[str]) -> List[str]:
        out = ["# 著录信息"]
        for ln in meta_lines:
            s = ln.strip()
            if not s:
                continue
            # 将顶层 "# (xx) ..." 降级为 "## (xx) ..."
            if s.startswith("# "):
                s = s[2:].strip()
            # 统一成二级小节
            if s.startswith("(") and ")" in s[:6]:
                out.append(f"## {s}")
            else:
                # 保留其余信息行
                out.append(f"## {s}")
        if abstract_lines:
            out.extend(self._trim_blank_edges(abstract_lines))
        else:
            out.append("（此处需替换：未能可靠提取摘要正文）")
        return out

    def _extract_title_text(self, lines: List[str], body_sections: Dict[str, List[str]]) -> Optional[str]:
        # 先找 (54) 名称
        for i, ln in enumerate(lines[:200]):
            if NAME_54_RE.search(ln):
                # 向下找首个非空且不是井号标题的行
                j = i + 1
                while j < len(lines):
                    cand = lines[j].strip()
                    if cand and not cand.lstrip().startswith("#"):
                        return cand
                    j += 1
                break

        # 次选：说明书中首个“独立一级标题”（不是“技术领域/背景技术/实用新型内容/发明内容/附图说明/具体实施方式”）
        # 通常是“居中”的题名，如“# 一种……”。
        known = {"技术领域", "背景技术", "实用新型内容", "发明内容", "附图说明", "说明书附图", "具体实施方式"}
        for blk_title in ["技术领域", "背景技术"]:
            # 在这些区块之前，若 body_lines（组装时）存在首个独立标题，可作为题名
            pass  # 已在 _extract_body_sections 统一抓取，故此处保守跳过

        # 直接从 body_sections 的原始块头扫描候选：
        # 由于 _extract_body_sections 是按一级 # 提取的，我们可以回到源文本找不属于 known 的一级标题
        # 构建一个集合帮助排除
        # 简化：在原文中抓取第一个一级 # 标题，若不在 known，则取它
        for ln in lines:
            m = HEADING_HASH_RE.match(ln)
            if m:
                title = m.group(1).strip().replace(" ", "")
                if title and (title not in known) and not title.startswith("("):
                    return m.group(1).strip()

        return None

    def _is_body_heading(self, line: str) -> bool:
        if not line.strip().startswith("#"):
            return False
        title = HEADING_HASH_RE.match(line).group(1).strip() if HEADING_HASH_RE.match(line) else ""
        # 说明书相关的常见标题集合
        return title in {"技术领域", "背景技术", "发明内容", "实用新型内容", "附图说明", "说明书附图", "具体实施方式"}

    def _extract_body_sections(self, body_lines: List[str]) -> Dict[str, List[str]]:
        """
        从 body_lines（摘要与权利要求书之后的剩余部分）中，按常见的一级 # 标题切分并抽取：
        - 技术领域
        - 背景技术
        - 发明内容 / 实用新型内容
        - 具体实施方式
        其余块（如“附图说明/说明书附图”）会提取但在最终组合时忽略。
        """
        sections: Dict[str, List[str]] = {}
        if not body_lines:
            return sections

        # 找到每个一级 # 的起止
        indices: List[Tuple[str, int, int]] = []  # (标题, start, end)
        curr_title = None
        curr_start = None
        for i, ln in enumerate(body_lines):
            m = HEADING_HASH_RE.match(ln)
            if m and ln.lstrip().startswith("# "):
                # 新块开始
                if curr_title is not None:
                    indices.append((curr_title, curr_start, i))
                curr_title = m.group(1).strip()
                curr_start = i + 1
        # 收尾
        if curr_title is not None:
            indices.append((curr_title, curr_start, len(body_lines)))

        # 提取内容
        for title, s, e in indices:
            content = self._trim_blank_edges(body_lines[s:e])
            if not content:
                continue
            # 只记录我们关心的块，其余可保留以备后用
            sections.setdefault(title, []).extend(content)

        return sections

    @staticmethod
    def _trim_blank_edges(lines: List[str]) -> List[str]:
        i, j = 0, len(lines)
        while i < j and not lines[i].strip():
            i += 1
        while j > i and not lines[j - 1].strip():
            j -= 1
        return lines[i:j]


# ------------------------------ 演示：用用户给出的示例文本运行并写出 /mnt/data/zz.md ------------------------------

raw = r"""# (19)中华人民共和国国家知识产权局

# (12)实用新型专利

(10)授权公告号CN207225508U(45)授权公告日2018.04.13

(21)申请号201721328994.5

(22)申请日2017.10.16

(73)专利权人杭州宇树科技有限公司地址310051浙江省杭州市滨江区聚业路26号金绣国际科技中心B座106室

(72)发明人王兴兴 杨知雨

(74)专利代理机构浙江翔隆专利事务所（普通合伙）33206

代理人许守金

(51)Int.Cl. B62D57/032(2006.01) G01L1/18(2006.01) G01C21/10(2006.01)

权利要求书1页 说明书4页 附图4页

# (54)实用新型名称

一种机器人足端结构

# (57)摘要
本实用新型公开了一种机器人足端结构，属于机器人设备技术领域。现有技术的足端结构较复杂，体积及重量大，内置的力传感器可靠性低。一种机器人足端结构，包括用于和机器人腿部连杆相连接的足基座总成、用于缓冲传递冲击力的足垫和保护罩总成，所述足基座总成内的传感部件上设有能产生较大变形的敏感部，所述敏感部由高屈服强度材料制成，所述敏感部表面粘贴有应变桥；所述足垫传递足受到的支撑力到足基座总成，使敏感部产生较大变形，从而通过应变桥检测到足端支撑力的大小。本实用新型集力传感器与足端为一体，零部件少，结构简单，能够有效减轻机器人足端的重量与体积，便于生产制造，力传感器过载能力强不易失效，制造成本低。

1. 一种机器人足端结构, 包括用于和机器人腿部连杆相连接的足基座总成 (2)、用于缓冲传递冲击力的足垫 (3); 其特征在于, 所述足基座总成 (2) 内的传感部件 (2-1) 上设有能产生较大变形的敏感部 (2-1-1), 所述敏感部 (2-1-1) 由高屈服强度材料制成; 所述足垫 (3) 传递足受到的支撑力到足基座总成 (2), 使敏感部 (2-1-1) 产生较大变形。

2. 如权利要求1所述的一种机器人足端结构, 其特征在于, 所述敏感部 (2-1-1) 位于传感部件 (2-1) 的中间位置, 所述传感部件 (2-1), 其端面形状为至少包含一个倾斜一定角度的字母Z和/或字母U和/或字母V形状结构。

3. 如权利要求2所述的一种机器人足端结构, 其特征在于, 所述高屈服强度材料为铝合金或碳钢或钛合金。

4. 如权利要求1所述的一种机器人足端结构, 其特征在于, 所述足基座总成 (2) 通过螺钉连接或粘接的方式连接一保护罩总成 (1)。

5. 如权利要求1所述的一种机器人足端结构, 其特征在于, 所述足垫 (3) 通过粘接或熔融方式与足基座总成 (2) 相连接。

6. 如权利要求1所述的一种机器人足端结构, 其特征在于, 所述敏感部 (2-1-1) 至少设有一对相互平行的平面区域 (2-1-1-1), 所述相互平行的平面区域 (2-1-1-1) 两侧至少连接一组用于感应形变的应变片 (2-2)。

7. 如权利要求1所述的一种机器人足端结构, 其特征在于, 所述传感部件 (2-1) 安装一测控单元 (2-4), 所述测控单元 (2-4) 设有电信号放大芯片和用于测量足端三轴加速度的传感器; 或设有三轴陀螺仪传感器, 用于检测足端三轴的角速度; 或设有一片微控制器MCU, 控制测控单元 (2-4) 上的, 电信号放大芯片、三轴加速度传感器芯片、三轴陀螺仪传感器芯片和发光件 (2-3), 并且微控制器通过通讯线与机器人的主控制器进行通讯。

8. 如权利要求1所述的一种机器人足端结构, 其特征在于, 所述足基座总成 (2) 或与足基座总成 (2) 相连接固定的保护罩总成 (1) 外设有一凹槽, 凹槽内安装一用于观测足端运动轨迹的发光件 (2-3)。

9. 如权利要求1所述的一种机器人足端结构, 其特征在于, 所述传感部件 (2-1) 和腿部连杆一体成型, 或通过焊接方式连接, 或通过螺钉连接的方式连接。
# 一种机器人足端结构

# 技术领域

[0001] 本实用新型涉及一种机器人足端结构，属于机器人设备技术领域。

# 背景技术

[0002] 现有技术的机器人足端结构，需要在实现对机器人腿部的准确支撑以外，还需要检测出足部的支撑力大小和碰撞等情况，同时，由于足位于机器人腿连杆的末端，腿连杆的转动惯量对足的重量很敏感，所以应尽量减小足的重量。

[0003] 中国专利(申请号201110451677.3)公开了一种多足步行机器人脚，包括多足步行机器人脚机械装置和多足步行机器人脚测力装置。多足步行机器人脚机械装置包括球形足端、减震测力部分和小腿脚连接件。减震测力部分由球形足端连接件、脚连接件、脚部套筒、减震压缩弹簧、压垫、固定螺钉组成。多足步行机器人脚测力装置由压力传感器和信号采集处理器组成。此技术方案采用球形足端结构，机器人行走灵活可靠；压缩弹簧减震装置减小了多足步行机器人在行走过程中与地面之间的冲击；采用压力传感器对多足步行机器人的脚力进行测量，降低了多足机器人脚力测量的难度和复杂性。但是此方案结构复杂，重量及体积较大。

# 实用新型内容

[0004] 针对现有技术的缺陷，本实用新型的目的在于提供一种不易损坏、失效，结构简单，重量轻，体积紧凑，便于生产制造的低成本机器人足端结构，进一步，能够准确测量足端受力的机器人足端结构。

[0005] 为实现上述目的，本实用新型的技术方案为：

[0006] 一种机器人足端结构，包括用于和机器人腿部连杆相连接的足基座总成、用于缓冲传递冲击力的足垫和用于保护足内部结构的保护罩总成，所述足基座总成内的传感部件上设有能产生较大变形的敏感部，所述敏感部由高屈服强度材料制成；所述足垫传递足受到的支撑力到足基座总成，使敏感部产生较大变形。所述机器人腿部连杆与传感部件可以一体制成，也可以通过焊接或者螺钉连接的方式相连接。本实用新型设置的传感部件的，结构简单，便于生产制造，不易失效，制造成本低。

[0007] 作为优选技术措施，所述敏感部位于传感部件的中间位置，所述传感部件，其端面形状为至少包含一个倾斜一定角度的字母Z和/或字母U和/或字母V形状结构。所述传感部件可由高屈服强度材料加工形成，优选，其截面形状至少包含一个倾斜一定角度的V型或U型结构。

[0008] 作为优选技术措施，所述高屈服强度材料为铝合金或碳钢或钛合金，可根据受力大小，工作强度，合理选择材料。

[0009] 作为优选技术措施，所述足基座总成通过螺钉连接或粘接的方式连接一保护罩总成，保护罩总成罩设足基座总成，能够有效保护足基座总成内部的部件，防止当足部受到外部冲击时内部精密脆弱零件受到损害。保护罩总成由高强度、抗冲击且耐磨材料制成，所述

材料可以是碳钢、工程塑料、铝合金、复合材料等，也可以是多种材料结合而成。

[0010] 作为优选技术措施，所述足垫通过粘接或熔融方式与足基座总成相连接，足垫能够有效减少支撑力的冲击，避免因为冲击过大损坏足基座总成内零部件，以及实现机器人足端与地面的摩擦接触。

[0011] 作为优选技术措施，所述敏感部至少设有一对相互平行的平面区域，所述相互平行的平面区域两侧至少连接一组用于感应形变的应变片。所述应变片把支撑力对敏感部产生的形变转化为自身电阻的改变量，应变片配合测控单元，产生与外部支撑力大小相对应的微弱电信号，测控单元上的电信号放大芯片采集此微弱电信号并进行放大和处理，并通过数字通讯接口传递给机器人主控制板。

[0012] 作为优选技术措施，所述传感部件安装一测控单元，所述测控单元设有电信号放大芯片和用于测量足端三轴加速度的传感器；或设有三轴陀螺仪传感器，用于检测足端三轴的角速度；或设有一片微控制器MCU，控制测控单元上的电信号放大芯片、三轴加速度传感器芯片、三轴陀螺仪传感器芯片和发光件，并且微控制器通过通讯线与机器人的主控制器进行通讯。通过三轴加速度芯片获得的数据可以计算获得足的姿态、足端碰撞、足端打滑等情况；三轴陀螺仪芯片，用于检测足端三轴的角速度，可以配合三轴加速度芯片进一步完善足端姿态的解算，或者单纯测量足端及其相连机器人腿部连杆的角速度。

[0013] 作为优选技术措施，所述足基座总成或保护罩总成外设有一凹槽，凹槽内安装一用于观测足端运动轨迹的发光件。所述发光件为LED灯，方便观测足端的运动轨迹，并且提高机器人的观赏性。

[0014] 与现有技术相比，本实用新型具有以下有益效果：

[0015] 本实用新型设置由高屈服强度材料制造而成的和具有特殊结构的敏感部，相比现有技术的机器人足端结构，所需的零部件少，结构简单，重量轻，便于生产制造，不易因外力过载而失效，制造成本低。

[0016] 进一步，本实用新型的应变片黏贴于敏感部上，能够准确感应传感部件的形变位移，把支撑力的大小转化为自身电阻变化的大小，进而实现对足端受力的测量并且测量精度高。

# 附图说明

[0017] 图1为本实用新型整体结构示意图；[0018] 图2为本实用新型剖视图；[0019] 图3为本实用新型爆炸视图；[0020] 图4为本实用新型局部爆炸视图；[0021] 图5为本实用新型部分结构示意图（不包括保护罩总成）；[0022] 图6为本实用新型设置发光件的结构示意图（不包括保护罩总成）。[0023] 附图标记说明：[0024] 1：保护罩总成，2：足基座总成，3：足垫，2- 1：传感部件，2- 2：应变片，2- 3：发光件，2- 4：测控单元，2- 5：足基座，2- 1- 1：敏感部，2- 1- 1- 1：平行平面区域。

# 具体实施方式

[0025] 为了使本实用新型的目的、技术方案及优点更加清楚明白，以下结合附图及实施例，对本实用新型进行进一步详细说明。应当理解，此处所描述的具体实施例仅仅用以解释本实用新型，并不用于限定本实用新型。

[0026] 相反，本实用新型涵盖任何由权利要求定义的在本实用新型的精髓和范围上做的替代、修改、等效方法以及方案。进一步，为了使公众对本实用新型有更好的了解，在下文对本实用新型的细节描述中，详尽描述了一些特定的细节部分。对本领域技术人员来说没有这些细节部分的描述也可以完全理解本实用新型。

[0027] 需要说明的是，当元件被称为“固定于”另一个元件，它可以直接在另一个元件上或者也可以存在居中的元件。当一个元件被认为是“连接”另一个元件，它可以是直接连接到另一个元件或者可能同时存在居中元件。相反，当元件被称作“直接在”另一元件“上”时，不存在中间元件。本文所使用的术语“上”、“下”以及类似的表述只是为了说明的目的。

[0028] 除非另有定义，本文所使用的所有技术和科学术语与属于本实用新型的技术领域的技术人员通常理解的含义相同。本文所使用的术语只是为了描述具体的实施例的目的，不是旨在限制本实用新型。本文所使用的术语“或/和”包括一个或多个相关的所列项目的任意的和所有的组合。

[0029] 如图1- 6所示，一种机器人足端结构，包括用于保护足内部结构的保护罩总成1、用于和机器人腿部连杆相连接的足基座总成2、用于和地面直接接触并缓冲传递冲击力的足垫3。

[0030] 所述足基座总成2内设有能产生较大变形用于测力的传感部件2- 1，所述传感部件2- 1由高屈服强度材料制成；所述足垫3传递足部与地面的支撑力到足基座总成2，足基座总成2再把力传递到与此足固定的机器人腿连杆上，在此过程中足基座总成2中的传感部件2- 1的敏感部2- 1- 1会产生较大变形。

[0031] 所述足基座总成2包括传感部件2- 1、应变片2- 2、发光件2- 3、测控单元2- 4、足基座2- 5。足基座2- 5为横放的圆柱结构，所述圆柱结构的柱面设有矩形截面槽，所述传感部件2- 1位于足基座2- 5的固定槽内，通过粘接或者螺钉连接的方式固定。所述足垫3通过粘接或者熔融的方式连接到足基座2- 5上。传感部件2- 1和足基座2- 5也可以是一个单独的整体零件。[0032] 所述传感部件2- 1形状近似于一个倾斜一定角度的字母“Z”，所述传感部件2- 1“Z”字的中部分是所述用于测力能产生较大变形的敏感部2- 1- 1，传感部件2- 1由高屈服强度材料制成。所述高屈服强度材料为铝合金或碳钢或钛合金。

[0033] 足基座总成2的外部设置一保护罩总成1，保护罩总成1通过螺钉连接或者粘接的方式固定在足基座总成2上，将足基座总成2内部的传感部件2- 1、应变片2- 2、测控单元2- 4以及足基座2- 5与传感部件2- 1相靠近的部分完全覆盖，防止当足部受到外部冲击时内部精密脆弱零件受到损害。保护罩总成1由高强度、抗冲击且耐磨材料制成，所述材料可以是碳钢、工程塑料、铝合金、复合材料等，也可以是多种材料结合而成。

[0034] 所述敏感部2- 1- 1至少设有一对相互平行的平面区域2- 1- 1- 1，所述相互平行的平面区域2- 1- 1- 1两侧粘接有至少一组用于感应形变的应变片2- 2。所述应变片2- 2把支撑力对敏感部2- 1- 1产生的形变转化为自身电阻的改变量，应变片2- 2配合测控单元2- 4，产生与外部支撑力大小相对应的微弱电信号，测控单元2- 4上的电信号放大芯片采集此微弱电信号并进行放大和处理，并通过数字通讯接口传递给机器人主控制板。

[0035] 所述测控单元2- 4安装于传感部件2- 1的弯折空间内, 所述测控单元2- 4设有电信号放大芯片和三轴加速度传感器, 或设有三轴陀螺仪传感器, 或设有一片微控制器MCU。三轴加速度芯片, 用于检测足端的三轴加速度, 通过三轴加速度数据或再附加配合三轴陀螺仪传感器可以计算获得足的姿态、足端碰撞、足端打滑等情况。三轴陀螺仪传感器, 用于检测足端三轴的角速度, 可以配合三轴加速度计进一步完善足端姿态的解算, 或者单纯测量足端及其相连机器人腿部连杆的角速度。微控制器MCU, 控制测控单元2- 4上的电信号放大芯片、三轴加速度传感器芯片、三轴陀螺仪传感器芯片和发光件2- 3, 并且微控制器通过通讯线与机器人的主控制器进行通讯。

[0036] 所述足基座2- 5的端面开设一凹槽, 所述凹槽通过粘接或者螺钉固定一用于观测足端运动轨迹的发光件2- 3, 所述发光件2- 3为LED灯, 方便观测足端的运动轨迹, 并提高观赏性。

[0037] 所述传感部件2- 1可以是与其相连的机器人腿部连杆的一部分, 是一个整体, 从而减少机器人腿部零件的数量, 提高机器人的集成度。

[0038] 以上所述仅为本实用新型的较佳实施例而已, 并不用以限制本实用新型, 凡在本实用新型的精神和原则之内所作的任何修改、等同替换和改进等, 均应包含在本实用新型的保护范围之内。
"""

normer = PatentMDNorm(raw, out_path=".log/zz.md")
out_path = normer.write()










