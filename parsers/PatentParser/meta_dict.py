from collections import OrderedDict
MetaDict_template = OrderedDict({
            "publ_no":"",          # (10) 申请公布号 or 授权公告号   <公开号>
            "publ_date":"",        # (45) 申请公布日 or 授权公告日
            "is_granted": bool,    # 是否授权，授权的话才会有专利号
            "patent_no":"",        # 专利号（由申请号生成）          <专利号> if is_granted is Ture
            "apply_no": '',        # (21) 申请号(不重要)            <申请号>
            "apply_time": "",      # (22) 申请日
            "title": "",           # (54) 专利标题（实用新型/发明 名称）
            "applicants": "",      # (54) 专利权人 申请人
            "address": "",         # "邮编 地址"
            "inventors": "",       # (71) 发明人
            "doc_type": "",        # (12) 发明专利申请 、实用新型专利
            "tech_field": "",      # # 技术领域 的正文（不含标题）
            
            "root_dir": "",        # 专利目录（绝对路径）
            "pdf_path": "",        # 原始 PDF 的绝对路径
            "fig_list": {},        # {"abs_im": ["摘要图", abs_path], "图1": ["描述", abs_path], ...}
        })
