# data_db/vector_db.py

from langchain_chroma import Chroma
from models.embedding_model import enbedding_model_cratefn


vector_store_cfg = {
    "persist_directory": "data_db/chroma_db/.hubei_vectdb",
    "collection_name": "local_knowledge",
}

def vector_store_cratefn(cfg=vector_store_cfg, load_mode="auto"):
    """ 加载指定路径下的 vector_db  """
    return Chroma(
        collection_name=cfg["collection_name"],
        embedding_function=enbedding_model_cratefn(),
        persist_directory=cfg["persist_directory"],
    )
    
    
    


