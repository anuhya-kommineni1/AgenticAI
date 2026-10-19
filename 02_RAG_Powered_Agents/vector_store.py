# Part 2: Chroma Vector Store & MMR
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def setup_vector_store(documents, embeddings_model):
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        collection_name="support_tickets"
    )
