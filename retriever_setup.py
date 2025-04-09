from pinecone_setup import conn_pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
import os

def get_retriever(index_name: str, bm25_params_path: str, alpha: float) -> PineconeHybridSearchRetriever:

    index = conn_pinecone(index_name=index_name)

    bm25_encoder = BM25Encoder()
    if os.path.exists(bm25_params_path):
        bm25_encoder.load(bm25_params_path)
    else:
        print(f"[Warning] BM25 params not found at '{bm25_params_path}'. Using default BM25 encoder.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, alpha=alpha, top_k=10, text_key="text"
    )

    return retriever