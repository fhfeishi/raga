# 用一个智能体工作流 结合 强大的LLM 来结构化提取 专利文件pdf（打印版、普通版）的信息

# 智能体 在代码这里的体现可能就是在于  function-calling  ， 能够自己决定是否使用工具/函数  -> 然后肯定还是我的机器来执行这个工具/函数
# 工作流的话，概念就宽泛一些， 由于我的llm可能并不支持 function-calling， 但是可以融入到工作流中来

# 输入文件：  mineru解析之后的 专利pdf文件.md  
#           - 专利pdf文件.md  存在于一个子文件夹中，附带的还有原始的PDF文件.pdf、images/ 等  

"""  
文件系统：
~ patent_root/
    + sub_dir/
        + *.md          # 解析之后的markdown  （对于打印版pdf的解析不够好）
        + *_origin.pdf  # 原pdf 
        + images/       # 抽出来的配图 (不存在遗漏)

-->
文件戳： 词条解析、需要配置一个 英文-中文对照的prompt、 文件类型， --> 写入原 sub_dir  metadata.json
正文  ：                                                         --> 写入原 sub_dir  textdata.md



结构：

Patent Document
    ├── Level 0: Metadata（结构化字段）
    │   ├── 专利标识
    │   ├── 申请信息
    │   ├── 权利人信息
    │   ├── 分类信息
    │   └── 摘要
    ├── Level 1: Claims（法律语言）
    │   ├── 独立权利要求
    │   └── 从属权利要求
    ├── Level 2: Specification（自然语言）
    │   ├── 技术领域
    │   ├── 背景技术
    │   ├── 发明内容
    │   ├── 附图说明
    │   └── 具体实施方式
    └── Level 3: Visual & Aux（辅助信息）
        ├── 附图   
        ├── 附图标记
        └── 页数信息

-->

Patent Document
    ├── Level 0: Metadata（结构化字段）   检索
    │   ├── 专利标识
    │   ├── 申请信息
    │   ├── 权利人信息
    │   ├── 分类信息
    │   └── 摘要
    ├── Level 1: Claims（法律语言）   +  
    │   ├── 独立权利要求
    │   └── 从属权利要求
    ├── Level 2: Specification（自然语言）  # 嵌入 + 检索
    │   ├── 技术领域
    │   ├── 背景技术
    │   ├── 发明内容
    │   ├── 附图说明
    │   └── 具体实施方式
    └── Level 3: Visual & Aux（辅助信息） （不嵌入、但是支持调用）
        └── 附图描述+附图 


--> patent_md_normalizer










"""

# .todo 


from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, step,  Workflow

# Event





