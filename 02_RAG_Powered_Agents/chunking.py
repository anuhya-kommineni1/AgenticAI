# Part 1: Chunking Strategies
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def run_chunking(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(documents)
