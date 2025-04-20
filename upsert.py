from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retriever_setup import get_retriever
from config import INDEX_NAME

def upsert(index_name: str, text: str, batch_size: int = 500) -> None:

    retriever = get_retriever(index_name=index_name, bm25_params_path="bm25_params.json", alpha=0.8)

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=150  # Helps maintain context
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    for i in range(0, len(chunks), batch_size):
        i_end = min(i+batch_size, len(chunks))
        batch = chunks[i:i_end]

        try:
            retriever.add_texts(batch)
            print(f"Successfully upserted batch {i // batch_size} with {len(batch)} documents")
        except Exception as e:
            print(f"Error upserting batch {i // batch_size}: {e}")

    print(f"Upserted {len(chunks)} chunks")

if __name__ == "__main__":
    document = "document.txt"

    # Load the document using langchain's TextLoader
    loader = TextLoader(document) 
    documents = loader.load()  # This loads the document into a list of Document objects

    text = str(documents[0].page_content)
    upsert(index_name=INDEX_NAME, text=text)
    print("Upsert completed.")