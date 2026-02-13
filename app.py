import streamlit as st
import os

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# Load API Key from Streamlit Secrets
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Mysoft Heaven AI", page_icon="🤖")
st.title("🤖 Mysoft Heaven (BD) Ltd AI Assistant")

# -------------------------
# RAG Setup
# -------------------------
@st.cache_resource
def initialize_rag():
    pdf_path = "data/AI Engineer Assessments.pdf"

    if not os.path.exists(pdf_path):
        return None

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # FREE embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = initialize_rag()

if vectorstore is None:
    st.error("❌ PDF not found. Put your PDF inside `data/` folder.")
    st.stop()

# -------------------------
# Groq LLM
# -------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_retries=2
)

# Strict system prompt
system_prompt = """
You are an AI assistant for Mysoft Heaven (BD) Ltd.
Answer ONLY based on provided documents.
If not found, say:
"I apologize, but I don't have information about that."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

combine_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_chain)

# -------------------------
# Chat History
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat UI
# -------------------------
user_input = st.chat_input("Ask about Mysoft Heaven...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")
