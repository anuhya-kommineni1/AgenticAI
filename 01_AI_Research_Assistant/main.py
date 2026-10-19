import os
import json
import uuid
from tavily import TavilyClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize API clients from environment variables
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def generate_sub_queries(topic: str) -> list[str]:
    """Generates 3 targeted sub-queries to gather comprehensive information."""
    prompt = f"""You are an expert researcher. Given the research topic below, break it down into 3 specific web search queries to cover key aspects, developments, and challenges.
    
    Topic: {topic}
    
    Return ONLY a JSON list of strings like: ["query 1", "query 2", "query 3"]"""
    
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except Exception:
        return [topic]

def fetch_search_results(queries: list[str]) -> list[dict]:
    """Executes Tavily web searches and aggregates page content."""
    documents = []
    for query in queries:
        print(f"Searching for: '{query}'...")
        response = tavily.search(query=query, search_depth="advanced", max_results=3)
        for result in response.get("results", []):
            documents.append({
                "url": result["url"],
                "title": result["title"],
                "content": result["content"]
            })
    return documents

def build_vector_store(documents: list[dict]):
    """Splits content into chunks and stores them in a fresh ChromaDB collection."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    langchain_docs = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            langchain_docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": doc["url"], "title": doc["title"]}
                )
            )
            
    unique_collection = f"research_docs_{uuid.uuid4().hex[:8]}"
    return Chroma.from_documents(
        documents=langchain_docs, 
        embedding=embeddings, 
        collection_name=unique_collection
    )

def retrieve_relevant_context(vectorstore, original_topic: str, k: int = 5) -> str:
    """Retrieves top-k most relevant passages for report generation."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(original_topic)
    
    formatted_context = ""
    for idx, doc in enumerate(retrieved_docs, 1):
        formatted_context += f"\n--- Source [{idx}]: {doc.metadata['title']} ({doc.metadata['source']}) ---\n"
        formatted_context += doc.page_content + "\n"
        
    return formatted_context

def run_research_assistant(topic: str):
    """Orchestrates the research process from query to report."""
    print(f"🚀 Starting Research Pipeline for: '{topic}'\n")
    
    print("Step 1/4: Decomposing topic into sub-queries...")
    sub_queries = generate_sub_queries(topic)
    
    print("Step 2/4: Searching the web with Tavily...")
    raw_docs = fetch_search_results(sub_queries)
    
    print("Step 3/4: Building RAG vector database...")
    vectorstore = build_vector_store(raw_docs)
    context = retrieve_relevant_context(vectorstore, topic)
    
    print("Step 4/4: Generating final report with OpenAI...")
    report_prompt = f"""You are a senior analyst. Write a comprehensive, well-structured research report in Markdown based ONLY on the context below.

Topic: {topic}

Retrieved Context:
{context}

Requirements:
1. Structure with: # Title, ## Executive Summary, ## Key Findings, ## Current Trends, and ## References.
2. Include in-text inline citations referencing the source URLs."""

    report_response = llm.invoke(report_prompt)
    return report_response.content

if __name__ == "__main__":
    report = run_research_assistant("Evolution of Computers")
    print(report)
