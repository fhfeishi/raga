from transformers import AutoTokenizer, AutoModelForCausalLM 
from typing import List
import os 
from huggingface_hub import constants

    

class HFModelsDownloader_Cache:
    def __init__(self, models_to_download: List[dict]) -> None:
        """
        初始化模型下载器 cache 版本
        :param models_to_download: 包含模型信息的字典列表，每个字典包含 'model_name' 键
        模型自动存到 c/user/.cache/huggingface  缓存下
        特点、可能会中断
        
        - 可以试试 huggingface-cli  
        $ pip install -U "huggingface_hub[cli]"  or pip install -U huggingface_hub
        # 1-a. 设置镜像源（Linux/macOS）  # Linux:~/.bashrc ~/.nahs_profile  # macOS: ~/.zshrc
        export HF_ENDPOINT=https://hf-mirror.com
        export HF_ENDPOINT=https://mirrors.tuna.tsinghua.edu.cn/hugging-face/
        # 1-b. windows
        ## cmd          
        set HF_ENDPOINT=https://mirrors.tuna.tsinghua.edu.cn/hugging-face/
        ## powershell  #  $PROFILE  or +系统环境变量+
        $env:HF_ENDPOINT="https://mirrors.tuna.tsinghua.edu.cn/hugging-face/"
        
        # 2. 下载模型（支持断点续传）
        huggingface-cli download --resume-download Qwen/Qwen3-Embedding-0.6B --local-dir ./temp/mineru_models/Qwen3-Embedding-0.6B
        hf download --resume-download Qwen/Qwen3-Embedding-0.6B --local-dir ./temp/mineru_models/Qwen3-Embedding-0.6B --local-dir-use-symlinks False
        # -> .cache  # 相对于 HFModelsDownloader_SnapShot 多一个.cache/
        """
        self.models_to_download = models_to_download
    
    def __call__(self):
        self._load_models()
    
    def _load_models(self) -> None:
        for model in self.models_to_download:
            model_name = model.get("model_name")
            local_dir = model.get("local_dir", None)
            
            # 加载模型和tokenizer     
            auto_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            auto_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            print(f"模型: {model_name} 已缓存到 {constants.HF_HUB_CACHE}")


from huggingface_hub import snapshot_download 
class HFModelsDownloader_SnapShot:
    def __init__(self, models_to_download: List[dict]) -> None:
        """
        初始化模型下载器 snapshot 版本
        :param models_to_download: 包含模型信息的字典列表，每个字典包含 'repo_id' 和 'local_dir' 两个键
        """
        self.models_to_download = models_to_download
         
    def __call__(self) -> None:
        self._load_models()
    
    def _load_models(self) -> None:
        for model in self.models_to_download:
            while True:  # 断点续传重试机制
                try:
                    print(f"开始下载模型: {model['repo_id']} 到目录: {model['local_dir']}")
                    snapshot_download(
                        repo_id=model["repo_id"],
                        local_dir=model["local_dir"],
                        resume_download=True,  # 启用断点续传
                        force_download=False,  # 避免重复下载已有文件
                        token=None,            # 如需访问私有模型，替换为你的 token
                    )
                    print(f"模型: {model['repo_id']} 下载到目录: {model['local_dir']} 完成！")
                    break
                except Exception as e:
                    print(f"下载失败: {e}, 重试中...")



if __name__ == '__main__':
    
    # from huggingface_hub import constants
    # HF 缓存根目录（Path 对象）
    # print("HF hub cache root:", constants.HF_HUB_CACHE)  
    # HF hub cache root: C:\Users\User\.cache\huggingface\hub
    
    
    # 受这些环境变量影响：
    #   HF_HOME                 -> 根目录（默认 ~/.cache/huggingface）
    #   或直接设置 HF_HUB_CACHE -> hub 子目录（默认 ~/.cache/huggingface/hub）
    # Transformers 旧变量（有时也会用到）：
    #   TRANSFORMERS_CACHE
    
    
    
    # 设置镜像源（国内加速）、任选其一
    os.environ["HF_ENDPOINT"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face/"
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    models_to_download = [
        {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",  # Embedding 模型 bge-m3 Qwen/Qwen3-Embedding-0.6B
            # "model_name":"Qwen/Qwen3-Reranker-0.6B"
        }
    ]
    downloader = HFModelsDownloader_Cache(models_to_download)
    downloader()  # 调用下载器
    
    # -> temp/mineru_models/Qwen3-Embedding-0.6B / models--Qwen--Qwen3-Embedding-0.6B
    #                                            / .lock/models--Qwen--Qwen3-Embedding-0.6B/
    
    # snapshot : download model-repo to local_dir
    # models_to_download = [
    #     {
    #         "repo_id": "Qwen/Qwen3-Embedding-0.6B",  # Embedding 模型 bge-m3 Qwen/Qwen3-Embedding-0.6B
    #         "local_dir": os.path.expanduser("temp/mineru_models/Qwen3-Embedding-0.6B"), # local_dir/xxxxx 可以是None
    #     }
    # ]
    # downloader = HFModelsDownloader_SnapShot(models_to_download)
    # downloader()  # 调用下载器
    
    # -> temp/mineru_models/Qwen3-Embedding-0.6B












