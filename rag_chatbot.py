"""
RAG Chatbot — Terminal Version
Ask questions from your PDF using Ollama + Pinecone
----------------------------------------------------
Prerequisites (run once):
    pip install langchain langchain-community langchain-ollama \
                langchain-pinecone pinecone-client pypdf langchain-text-splitters

Make sure Ollama is running:
    ollama serve

Usage:
    python rag_chatbot.py
"""

import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# ── ✏️  Change these to match your setup ──────────────────────────────────────

EMBEDDING_MODEL  = "embeddinggemma"               # embedding model pulled in Ollama
LLM_MODEL        = "tinyllama"                    # LLM model pulled in Ollama
OLLAMA_URL       = "http://localhost:11434"        # Ollama default URL

PINECONE_API_KEY = "pcsk_Qe2SL_5cQP7UcSSVGS1RK2syXyFfyQ4GgtwRranCkkFaFu2A7jtC1bxD47NdYDMrFzvyn"
PINECONE_INDEX   = "rag"                    # must already exist in Pinecone

TOP_K            = 5                              # number of chunks to retrieve per question

# ─────────────────────────────────────────────────────────────────────────────


def load_components():
    """Load Pinecone vectorstore, retriever, LLM, and RAG chain."""

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    print("🔗 Connecting to Pinecone ...")
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embedding_model,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    print("🤖 Loading LLM ...")
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.1)

    rag_chain = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Use only the context below to answer the question.
If the answer is not in the context, say: "I don't know based on this document."
Do not make up any information.

Context from the document:
{context}

Question:
{input}

Answer:
""") | llm

    return retriever, rag_chain


def chat(retriever, rag_chain, question: str) -> str:
    """Retrieve relevant chunks and generate an answer."""
    chunks = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in chunks)
    answer = rag_chain.invoke({"context": context, "input": question})
    pages = set(str(doc.metadata.get("page", "?")) for doc in chunks)
    return answer, pages


def main():
    print("\n" + "=" * 60)
    print("  RAG Chatbot  |  Ollama + Pinecone")
    print("=" * 60)

    retriever, rag_chain = load_components()

    print("\n✅ Ready! Type your question below.")
    print("   Type  'exit'  or  'quit'  to stop.\n")
    print("-" * 60)

    while True:
        try:
            question = input("\n❓ You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye!")
            break

        print("\n⏳ Thinking ...\n")
        answer, pages = chat(retriever, rag_chain, question)

        print(f"🤖 Bot: {answer}")
        print(f"\n📄 Pages referenced: {', '.join(sorted(pages))}")
        print("-" * 60)


if __name__ == "__main__":
    main()
