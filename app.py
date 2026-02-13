import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# API KEY
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY in Streamlit Secrets")
    st.stop()

# -------------------------
# LOAD PDF
# -------------------------
def load_vectorstore():
    if not os.path.exists("data"):
        return None
    
    loader = PyPDFLoader("data/AI Engineer Assessments.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vectorstore = load_vectorstore()
if vectorstore is None:
    st.error("No PDF found in data folder")
    st.stop()

# -------------------------
# LLM
# -------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")

# -------------------------
# Prompt
# -------------------------
system_prompt = """
You are Mysoft Heaven AI assistant.
Answer ONLY from company documents.
If not found, say: I don't have information about that.
Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# -------------------------
# Streamlit UI
# -------------------------
st.title("🤖 Mysoft Heaven (BD) Ltd AI Assistant")

query = st.chat_input("Ask about Mysoft Heaven...")

if query:
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(final_prompt)

    st.chat_message("user").write(query)
    st.chat_message("assistant").write(response.content)

