import os

from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OllamaEmbeddings(model="all-minilm")
# embeddings = OllamaEmbeddings(model="llama3")
text = "Hello world"
single_vector = embeddings.embed_query(text)
print(len(single_vector))


def run():
    print("Loading raw documents...")
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"> Loaded {len(raw_documents)} raw documents.")

    print("Loading documents....")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https://")
        doc.metadata.update({"source": new_url})
    print(f">Loaded {len(documents)} documents")

    print("Adding documents to Pinecone...")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=os.getenv("PINECONE_INDEX")
    )
    print("> Added documents to Pinecone.")


if __name__ == "__main__":
    run()
    print("Ingestion done! ✨ 🍰 ✨")
