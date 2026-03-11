from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def retrieval_pipeline(user_query):
    # 1. Initialize Embeddings & Load DB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # 2. Initialize Local LLM
    llm = ChatOllama(model="llama3", temperature=0)

    # 3. Define the Prompt Template
    # This gives you total control over how the AI behaves
    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Define the Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 5. Build the LCEL Chain
    # This replaces the old RetrievalQA.from_chain_type
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 6. Get the answer
    print(f"\nQuestion: {user_query}")
    print("-" * 30)

    return chain.invoke(user_query)


if __name__ == "__main__":
    query = "Who is the founder of  Google ?"
    answer = retrieval_pipeline(query)
    print(f"AI Answer:\n{answer}")
