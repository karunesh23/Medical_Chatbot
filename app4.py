import streamlit as st
import time
from rag1 import load_components, chat

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
    color:white;
}

/* Main title */
.title{
    text-align:center;
    font-size:38px;
    font-weight:bold;
    color:#4EA8FF;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#CCCCCC;
    margin-bottom:25px;
}

/* Chat bubble */
.user-box{
    background:#1E293B;
    padding:15px;
    border-radius:12px;
    margin-bottom:10px;
    border-left:4px solid #4EA8FF;
}

.bot-box{
    background:#111827;
    padding:15px;
    border-radius:12px;
    margin-bottom:15px;
    border-left:4px solid #22C55E;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
}

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:10px;
    height:45px;
    font-weight:bold;
}

/* Footer */
.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================

with st.sidebar:

    st.title("🩺 Medical AI")

    st.divider()

    st.subheader("🤖 Model")
    st.success("Qwen2.5 : 7B")

    st.subheader("📚 Embedding")
    st.info("nomic-embed-text")

    st.subheader("🗄 Vector DB")
    st.warning("Pinecone")

    st.subheader("📄 Knowledge Base")
    st.write("Medical PDF")

    st.subheader("🔍 Retrieval")
    st.write("Similarity Search")

    st.divider()

    st.subheader("⚙ Settings")

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.0,
        0.1
    )

    top_k = st.slider(
        "Top K",
        1,
        10,
        4
    )

    show_sources = st.checkbox(
        "Show Sources",
        value=True
    )

    st.divider()

    if st.button("🧹 Clear Chat"):

        st.session_state.messages = []
        st.rerun()

# ==============================
# HEADER
# ==============================

st.markdown(
    "<div class='title'>🩺 Medical AI Assistant</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Retrieval-Augmented Generation (RAG) using Pinecone + Ollama + Qwen2.5</div>",
    unsafe_allow_html=True
)

# ==============================
# SESSION STATE
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""


# ==============================
# CHAT HISTORY
# ==============================

for msg in st.session_state.messages:

    if msg["role"]=="user":

        st.markdown(
            f"""
<div class="user-box">

👤 <b>You</b>

{msg["content"]}

</div>
""",
unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
<div class="bot-box">

🤖 <b>Medical AI</b>

{msg["content"]}

</div>
""",
unsafe_allow_html=True
        )

# ==============================
# CHAT INPUT
# ==============================

question = st.chat_input(
    "Ask your medical question..."
)

# ==========================================
# IMPORT YOUR RAG FUNCTIONS
# ==========================================


# Load only once
if st.session_state.vectorstore is None:

    with st.spinner("Loading AI Model..."):

        vectorstore, chain = load_components()

        st.session_state.vectorstore = vectorstore
        st.session_state.chain = chain

        st.success("✅ Medical AI loaded successfully")


# ==========================================
# ASK QUESTION
# ==========================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    start = time.time()

    with st.spinner("🔍 Searching document and generating answer..."):

        answer, pages = chat(
            st.session_state.vectorstore,
            st.session_state.chain,
            question
        )

    end = time.time()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)

        if show_sources and pages:

            st.success(
                "📄 Source Pages : " + ", ".join(pages)
            )

        st.caption(
            f"⏱ Response Time : {end-start:.2f} sec"
        )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
"""
<div class='footer'>

🩺 Medical AI Assistant

Powered by

<b>LangChain</b> •
<b>Pinecone</b> •
<b>Ollama</b> •
<b>Qwen2.5</b>

</div>
""",
unsafe_allow_html=True
)