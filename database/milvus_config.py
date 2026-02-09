from langchain_milvus import Milvus
from models.gemini_emb_model import embeddings

URI = "http://localhost:19530"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI, "token": "root:Milvus", "db_name": "milvus_demo"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    consistency_level="Strong",
    collection_name="agentic_rag_docs",
    drop_old=False,  
)