import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# ---------------- CONFIG ----------------

PDF_FILE = "text_medical.pdf"

EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"

PINECONE_API_KEY = "pcsk_eVBrT_KoYuHnxKcKPADABmZ76kUZBJYR3Y7ft9AUp7uADBmsJABBiiXxkzCjzrXZp6vut"

PINECONE_INDEX = "rag"

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

EMBEDDING_DIM = 768

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ----------------------------------------


def load_pdf():

    print("=" * 60)
    print("Loading PDF...")
    print("=" * 60)

    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()

    print(f"Total Pages : {len(documents)}")

    return documents


def split_pdf(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks : {len(chunks)}")

    return chunks


def remove_duplicates(chunks):

    print("Removing duplicate chunks...")

    seen = set()
    unique_chunks = []

    for chunk in chunks:

        text = chunk.page_content.strip()

        if text in seen:
            continue

        seen.add(text)
        unique_chunks.append(chunk)

    print(f"Unique Chunks : {len(unique_chunks)}")

    return unique_chunks


def create_embeddings():

    print("Loading Embedding Model...")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )

    vector = embeddings.embed_query("hello")

    print(f"Embedding Dimension : {len(vector)}")

    return embeddings


def create_index():

    pc = Pinecone(api_key=PINECONE_API_KEY)

    indexes = pc.list_indexes().names()

    if PINECONE_INDEX not in indexes:

        print("Creating Pinecone Index...")

        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

        print("Index Created.")

    else:

        print("Using Existing Index.")

    return pc


def upload(chunks, embeddings):

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    print("Uploading Chunks to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX
    )

    print("=" * 60)
    print("Upload Completed Successfully")
    print("=" * 60)


def main():

    documents = load_pdf()

    chunks = split_pdf(documents)

    chunks = remove_duplicates(chunks)

    embeddings = create_embeddings()

    create_index()

    upload(chunks, embeddings)


if __name__ == "__main__":
    main()