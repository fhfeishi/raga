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
正文  ：                                                      --> 写入原 sub_dir  textdata.md
"""

# .todo 


from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, step,  Workflow

# Event





