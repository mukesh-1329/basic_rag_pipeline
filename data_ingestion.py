from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def main():
    file_path = "Google.pdf"
    #1 loading
    print("loading file: " )
    loader = PyPDFLoader(file_path)
    data =loader.load()

    #2.chunking
    print("chunking")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    #3.embedding and storing in database
    local_embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=local_embeddings,
        persist_directory="./chroma_db"
    )

    print(f"Ingested {len(chunks)} chunks into the vector store.")



if __name__ == "__main__":
    main()


# for the list of chunks.
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def list_all_chunks():
    # 1. Initialize the same embedding model used during ingestion
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 2. Load the existing database from the directory
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=local_embeddings
    )

    # 3. Get all data from the collection
    # .get() returns a dictionary containing 'documents', 'metadatas', and 'ids'
    data = vector_db.get()
    all_documents = data['documents']
    all_metadatas = data['metadatas']

    print(f"--- Found {len(all_documents)} chunks in the database ---\n")

    # 4. Loop through and print the chunks
    for i, (text, meta) in enumerate(zip(all_documents, all_metadatas)):
        # Getting the page number from metadata if available
        page_num = meta.get('page', 'Unknown')

        print(f"ID: {i + 1} | Source: {meta.get('source')} | Page: {page_num}")
        print(f"Content snippet: {text[:150]}...")  # Printing first 150 chars
        print("-" * 50)


if __name__ == "__main__":
    list_all_chunks()