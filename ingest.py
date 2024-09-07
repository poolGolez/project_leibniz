from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    print(f"Loaded {len(documents)} documents")
    print("Done.")

if __name__ == "__main__":
    run()