from pinecone_setup import conn_pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

def get_retriever(index_name: str, bm25_params_path: str, alpha: float) -> PineconeHybridSearchRetriever:

    index = conn_pinecone(index_name=index_name)

    bm25_encoder = BM25Encoder()
    bm25_encoder.load(bm25_params_path)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, alpha=alpha, top_k=10, text_key="text"
    )

    return retriever