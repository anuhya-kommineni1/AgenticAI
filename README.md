# 🔬 Autonomous AI Research Assistant

An intelligent, multi-step research agent that breaks down broad topics into targeted sub-queries, searches the web via Tavily API, indexes content in a Chroma vector database, and generates structured research reports with inline citations using RAG and OpenAI.

## 🚀 Key Features
- **Query Expansion**: Automatically decomposes complex research topics into 3 targeted web search queries.
- **Web Scraping & Aggregation**: Fetches clean, parsed webpage text using Tavily Search API.
- **In-Memory RAG Architecture**: Chunks text and generates vector embeddings using `OpenAIEmbeddings` (`text-embedding-3-small`) stored in `ChromaDB`.
- **Structured Synthesis**: Synthesizes retrieved context into professional Markdown reports complete with executive summaries, key findings, and citations.

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **LLM Engine**: OpenAI GPT-4o-mini (`langchain-openai`)
- **Search Engine API**: Tavily Search API (`tavily-python`)
- **Vector Database**: ChromaDB (`chromadb`)
- **Framework**: LangChain (`langchain-community`, `langchain-core`)

## 🔑 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/anuhya-kommineni1/AgenticAI.git](https://github.com/anuhya-kommineni1/AgenticAI.git)
   cd AgenticAI
