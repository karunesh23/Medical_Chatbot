import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# ==========================
# CONFIGURATION
# ==========================

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

OLLAMA_URL = "http://localhost:11434"

PINECONE_API_KEY = "pcsk_eVBrT_KoYuHnxKcKPADABmZ76kUZBJYR3Y7ft9AUp7uADBmsJABBiiXxkzCjzrXZp6vut"
PINECONE_INDEX = "rag"

TOP_K = 4
SIMILARITY_THRESHOLD = 0.55

# ==========================
# LOAD COMPONENTS
# ==========================


def load_components():

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    print("=" * 60)
    print("Connecting to Pinecone...")
    print("=" * 60)

    embedding_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embedding_model
    )

    print("Loading Qwen 2.5 Model...")

    llm = OllamaLLM(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
    num_predict=400
)

    prompt = ChatPromptTemplate.from_template("""
You are an expert medical assistant.

Answer ONLY using the provided medical context.

Rules:
- Use ONLY the provided context.
- Never use outside knowledge.
- Never hallucinate.
- Never mention context, documents, retrieval, embeddings, Pinecone, LangChain, or RAG.
- If the retrieved context is not relevant, reply ONLY with:

"I don't have enough information in my medical knowledge base."

- Never mix an answer with the fallback message.
- Skip any section not supported by the context.

Formatting Rules:
- Use Markdown.
- Use headings with ##.
- Use bullet points.
- Keep answers concise and professional.

Response Format:

## Definition
A short definition (if available).

## Symptoms
• item
• item

## Causes
• item

## Diagnosis
• item

## Treatment
• item

## Prevention
• item

## Complications
• item

Context:
{context}

Question:
{input}

Answer:
""")

    chain = prompt | llm

    return vectorstore, chain


# ==========================
# RETRIEVE CONTEXT
# ==========================

# ==========================
# RETRIEVE CONTEXT
# ==========================

def retrieve_context(vectorstore, question):

    print("Vectorstore:", vectorstore)
    print("Question:", question)

    try:
        results = vectorstore.similarity_search_with_score(
            question,
            k=TOP_K
        )

        if not results:
            return None, None

        print("Results:", len(results))

        best_score = results[0][1]
        print("Best Score:", best_score)

        # Similarity threshold
        if best_score < SIMILARITY_THRESHOLD:
            return None, None

    except Exception as e:
        print("ERROR:", e)
        return None, None

    docs = []

    for doc, score in results:
        print(f"Score: {score:.4f}")
        docs.append(doc)

    if len(docs) == 0:
        return None, None

    unique_chunks = set()
    final_docs = []

    for doc in docs:
        text = doc.page_content.strip()

        if text in unique_chunks:
            continue

        unique_chunks.add(text)
        final_docs.append(doc)

    context_parts = []

    for doc in final_docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")

        context_parts.append(
            f"""
----------------------------------------
Source : {source}
Page   : {page}

Content:
{doc.page_content}
----------------------------------------
"""
        )

    context = "\n".join(context_parts)

    return context, final_docs


# ==========================
# CHAT FUNCTION
# ==========================

def chat(vectorstore, chain, question):

    context, docs = retrieve_context(
        vectorstore,
        question
    )

    if context is None:
        return (
            "I don't have enough information in my medical knowledge base.",
            []
        )

    answer = chain.invoke(
    {
        "context": context,
        "input": question
    }
).strip()

    pages = []

    for doc in docs:

        page = doc.metadata.get("page")

        if page is not None:
            pages.append(str(page + 1))

    pages = sorted(list(set(pages)))

    return answer, pages

# ==========================
# MAIN PROGRAM
# ==========================

def main():

    print("=" * 70)
    print("        RAG Chatbot (Qwen2.5 + Pinecone + Ollama)")
    print("=" * 70)

    vectorstore, chain = load_components()

    print("\n✅ Chatbot Ready")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:

        try:

            question = input("❓ You : ").strip()

        except KeyboardInterrupt:

            print("\nGoodbye!")
            break

        if question == "":
            continue

        if question.lower() in ["exit", "quit", "q"]:

            print("\nGoodbye!")
            break

        print("\nSearching documents...")
        print("Generating answer...\n")

        answer, pages = chat(
            vectorstore,
            chain,
            question
        )

        print("=" * 70)
        print("🤖 Answer\n")
        print(answer)

        if pages:

            print("\n📄 Source Pages :", ", ".join(pages))

        print("=" * 70)


# ==========================
# ENTRY POINT
# ==========================

if __name__ == "__main__":
    main()