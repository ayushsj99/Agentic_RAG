from langchain_milvus import Milvus
from models.gemini_emb_model import embeddings

URI="./milvus_example.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)