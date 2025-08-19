# models/chat_model.py

from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline


import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


chat_model_config = {
        "type": "huggingface",  # "huggingface" 或 "ollama"
        "huggingface_model": "Qwen/Qwen3-1.7B",
        "ollama_model": "qwen3:latest",
        "ollama_base_url": "http://localhost:11434",
        "temperature": 0.2,
        "max_new_tokens": 256,
        "context_window": 1000,
    
}

def chat_model_cratefn(llm_cfg=chat_model_config, load_mode='auto') -> BaseChatModel:
    """ load_mode: aoto.  aoto-load from c/usr/.cache """
    if llm_cfg["type"] == "huggingface":
        model_name = llm_cfg["huggingface_model"]
        logger.info(f"🔧 使用 HuggingFace LLM: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=llm_cfg["max_new_tokens"],
            temperature=llm_cfg["temperature"],
            do_sample=True,
            top_p=0.95,
            top_k=9,
            repetition_penalty=1.1,
            return_full_text=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFace(llm=hf_pipeline, model_id=model_name)

    elif llm_cfg["type"] == "ollama":
        logger.info(f"🔧 使用 Ollama LLM: {llm_cfg['ollama_model']}")
        return ChatOllama(
            base_url=llm_cfg["ollama_base_url"],
            model=llm_cfg["ollama_model"],
            temperature=llm_cfg["temperature"],
            num_ctx=llm_cfg["context_window"],
            num_predict=llm_cfg["max_new_tokens"],
            streaming=True, # 确保启用
            # options={...} 可扩展
        )
    else:
        raise ValueError(f"不支持的 LLM 类型: {llm_cfg['type']}")



        
        
        