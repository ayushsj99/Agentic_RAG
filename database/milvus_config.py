from langchain_milvus import BM25BuiltInFunction, Milvus
from models.gemini_emb_model import embeddings
from models.ollama_emb import ollama_embeddings

URI = "http://localhost:19530"

vectorstore = Milvus(
    embedding_function=ollama_embeddings,
    connection_args={"uri": URI, "token": "root:Milvus", "db_name": "milvus_demo"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    consistency_level="Strong",
    drop_old=False,  # set to True if seeking to drop the collection with that name if it exists
)