import streamlit as st
import time
import os
import html
from dotenv import load_dotenv
from datetime import datetime

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="wide")

# ------------------ THEME ------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ------------------ RAG SETUP ------------------
@st.cache_resource
def get_components(model_name):
    embedding = OllamaEmbeddings(model="embeddinggemma", base_url="http://localhost:11434")
    vectorstore = PineconeVectorStore(index_name="rag", embedding=embedding)
    
    llm = OllamaLLM(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.1,
        num_predict=500,
    )
    return vectorstore, llm, embedding

# Default
vectorstore, llm, embedding = get_components("tinyllama")

# ------------------ PROMPT ------------------
def get_prompt(medical_mode):
    if medical_mode:
        return ChatPromptTemplate.from_template("""
You are a precise medical assistant. Answer the question directly using only the context.
Provide clear symptoms, advice, or explanations. Use bullet points when helpful.
Never give medical advice as a substitute for professional care.

Context: {context}
Question: {input}
Answer:""")
    else:
        return ChatPromptTemplate.from_template("""
Answer ONLY from the given context. Be direct and concise.

Context: {context}
Question: {input}
Answer:""")

# ------------------ SIDEBAR TABS ------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    tab1, tab2, tab3 = st.tabs(["Models", "Documents", "Settings"])

    with tab1:
        model_options = ["tinyllama", "llama3.1", "mistral", "gemma2"]
        selected_model = st.selectbox("LLM Model", model_options, index=0)
        if selected_model != st.session_state.get("current_model"):
            st.session_state.current_model = selected_model
            vectorstore, llm, embedding = get_components(selected_model)
            st.success(f"Switched to {selected_model}")

    with tab2:
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            with st.spinner("Processing & Indexing PDF..."):
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                
                vectorstore.add_documents(splits)
                st.success(f"✅ Indexed {len(splits)} chunks from {uploaded_file.name}")
                os.remove(temp_path)

    with tab3:
        medical_mode = st.toggle("🩺 Medical Mode", value=True)
        if st.button("🗑 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ------------------ MAIN UI ------------------
st.markdown(f"""
<div style="text-align:center; padding:20px 0;">
    <h1 style="background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem;">
        🧠 AI Research Assistant
    </h1>
    <p>RAG-Powered • Real-time • Medical Ready</p>
</div>
""", unsafe_allow_html=True)

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"""
                    **{s['source']} — Page {s['page']}**  
                    🎯 Similarity: `{s['similarity']*100:.1f}%`  
                    <span style="color:#94a3b8; font-size:0.9em;">{s['snippet']}</span>
                    """, unsafe_allow_html=True)

# ------------------ INPUT & RESPONSE ------------------
if prompt := st.chat_input("Ask a medical or research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            
            results = vectorstore.similarity_search_with_score(prompt, k=5)
            context = "\n\n".join([doc.page_content for doc, _ in results])

            chain = get_prompt(medical_mode) | llm
            response = chain.invoke({"context": context, "input": prompt})
            
            elapsed = time.time() - start

            # Display response with typing effect
            placeholder = st.empty()
            full_response = ""
            for word in str(response).split():
                full_response += word + " "
                placeholder.write(full_response + "▌")
                time.sleep(0.015)
            placeholder.write(full_response)

        # Sources with Confidence
        sources = []
        for doc, score in results[:4]:
            sources.append({
                "source": doc.metadata.get("source", "Document"),
                "page": int(doc.metadata.get("page", 0)) + 1,
                "similarity": float(score),
                "snippet": doc.page_content[:180] + "..."
            })

        if sources:
            with st.expander("📚 Sources Used", expanded=True):
                for s in sources:
                    col1, col2 = st.columns([4,1])
                    with col1:
                        st.markdown(f"**{s['source']} — Page {s['page']}**")
                        st.caption(s['snippet'])
                    with col2:
                        st.progress(min(s['similarity'], 1.0))
                        st.caption(f"{s['similarity']*100:.1f}%")

        # Feedback
        col1, col2, col3 = st.columns([1,1,4])
        with col1:
            if st.button("👍", key=f"like_{len(st.session_state.messages)}"):
                st.toast("Thank you for your feedback!")
        with col2:
            if st.button("👎", key=f"dislike_{len(st.session_state.messages)}"):
                st.toast("We'll improve based on this.")

        # Save to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "latency": elapsed
        })

# ------------------ EXPORT ------------------
if st.session_state.messages:
    if st.sidebar.button("📤 Export Chat as Markdown"):
        md_content = "# AI Research Assistant Chat\n\n"
        for m in st.session_state.messages:
            role = "User" if m["role"] == "user" else "Assistant"
            md_content += f"### {role}\n{m['content']}\n\n"
        
        st.download_button(
            label="Download Chat.md",
            data=md_content,
            file_name=f"research_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )