import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def ask(query: str, history: List[Dict[str, Any]] = []):
    embeddings = OllamaEmbeddings(model="all-minilm")
    vector_store = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
    chat = ChatOllama(model="llama3")

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=vector_store.as_retriever(),
        prompt=rephrase_prompt)

    qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, qa_chat_prompt)
    qa = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)

    result = qa.invoke({
        "input": query,
        "chat_history": history
    })
    return result['answer']


if __name__ == "__main__":
    print("Inside RAG. ")
    # question = "Should I do embeddings for a question answer chatbot?"
    question = "What should be the dimensions for my text embeddings?"
    answer = ask(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("Query done! ✨ 🍰 ✨")
