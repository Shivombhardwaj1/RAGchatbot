import os
import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM

# ── Config ────────────────────────────────────────────────
UPLOAD_FOLDER = "./uploaded_docs"
DB_FOLDER = "./chroma_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)


# ── Load Models ───────────────────────────────────────────
@st.cache_resource
def get_llm():

    return OllamaLLM(model="mistral")


@st.cache_resource
def get_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Extract PDF Text ──────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


# ── Build Vector Store ────────────────────────────────────
def build_vector_store(texts: list, filenames: list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    all_metadata = []

    for text, filename in zip(texts, filenames):
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        all_metadata.extend([{"source": filename}] * len(chunks))

    return Chroma.from_texts(
        texts=all_chunks,
        embedding=get_embeddings(),
        metadatas=all_metadata,
        persist_directory=DB_FOLDER,
    )


# ── Load Existing Vector Store ────────────────────────────
def load_vector_store():
    return Chroma(persist_directory=DB_FOLDER, embedding_function=get_embeddings())


# ── Get Answer (No chains — direct Ollama call) ───────────
def get_answer(question: str):
    llm = get_llm()

    # ── PATH A: PDF uploaded → RAG ────────────────────────
    if "vector_store" in st.session_state:

        # search ChromaDB for relevant chunks
        results = st.session_state["vector_store"].similarity_search(question, k=4)

        # join chunks into context
        context = "\n\n".join(doc.page_content for doc in results)

        # collect source filenames
        sources = list({doc.metadata["source"] for doc in results})

        # build chat history string
        history = ""
        for msg in st.session_state["messages"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"

        # build RAG prompt manually
        prompt = f"""You are a helpful assistant.
Use the context below to answer the question.
If answer is not in context say you don't know.

Previous conversation:
{history}

Context from documents:
{context}

Question: {question}
Answer:"""

        answer = llm.invoke(prompt)
        return answer, sources

    # ── PATH B: No PDF → general chat ────────────────────
    else:
        # build chat history string
        history = ""
        for msg in st.session_state["messages"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"

        # simple chat prompt
        prompt = f"""You are a helpful assistant.

Previous conversation:
{history}

User: {question}
Assistant:"""

        answer = llm.invoke(prompt)
        return answer, []


# ── Send Message Helper ───────────────────────────────────
def send_message(question: str):
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(question)
            st.markdown(answer)
            if sources:
                st.caption("� Sources: " + ", ".join(sources))

    st.session_state["messages"].append({"role": "assistant", "content": answer})


# ── Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 LangChain RAG Chatbot")
st.caption("Chat freely — or upload PDFs to ask questions from them!")

# ── Session State ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# auto load saved ChromaDB on startup
if "vector_store" not in st.session_state:
    if os.path.exists(DB_FOLDER) and os.listdir(DB_FOLDER):
        st.session_state["vector_store"] = load_vector_store()
        st.sidebar.success("⚡ Loaded saved documents!")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDFs")

    # show currently loaded files
    if "vector_store" in st.session_state:
        loaded = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
        if loaded:
            st.markdown("**Currently loaded:**")
            for f in loaded:
                st.markdown(f"- 📄 {f}")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        texts, names = [], []
        for file in uploaded_files:
            path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            texts.append(extract_text(path))
            names.append(file.name)

        with st.spinner("Indexing PDFs..."):
            st.session_state["vector_store"] = build_vector_store(texts, names)
        st.success(f"✅ {len(uploaded_files)} PDF(s) indexed!")

    if st.button("🗑️ Clear PDFs"):
        st.session_state.pop("vector_store", None)
        import shutil

        if os.path.exists(DB_FOLDER):
            shutil.rmtree(DB_FOLDER)
            os.makedirs(DB_FOLDER)
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        st.success("Cleared!")
        st.rerun()

    if st.button("🔄 Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

# ── Chat History ──────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────
if prompt := st.chat_input("Ask anything..."):
    send_message(prompt)
