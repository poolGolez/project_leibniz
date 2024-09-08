import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def ask(query: str):
    embeddings = OllamaEmbeddings(model="all-minilm")
    vector_store = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
    chat = ChatOllama(model="llama3")

    qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, qa_chat_prompt)
    qa = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=stuff_documents_chain)

    result = qa.invoke({"input": query})
    return result['answer']


if __name__ == "__main__":
    print("Inside RAG. ")
    # question = "Should I do embeddings for a question answer chatbot?"
    question = "What should be the dimensions for my text embeddings?"
    answer = ask(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("Query done! ✨ 🍰 ✨")
