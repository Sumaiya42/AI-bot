import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------------------------
# API Key
# -------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY in Streamlit Secrets")
    st.stop()


# -------------------------------------------------------------------
# Load PDFs and Create Vector Store
# -------------------------------------------------------------------
def load_vectorstore():
    if not os.path.exists("data"):
        return None

    docs = []
    files = [
        "data/AI Engineer Assessments.pdf",
        "data/Mysoftheaven-Profile 2026.pdf",
    ]

    for file in files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


vectorstore = load_vectorstore()
if vectorstore is None:
    st.error("No PDF found in data folder")
    st.stop()


# -------------------------------------------------------------------
# LLM Setup
# -------------------------------------------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")


# -------------------------------------------------------------------
# Manual Conversation Memory (NO LANGCHAIN IMPORT)
# -------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------------------------------------------------
# Prompt Template
# -------------------------------------------------------------------
system_prompt = """
You are Mysoft Heaven AI assistant.
Answer ONLY from company documents.

Conversation History:
{history}

Context from documents:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Mysoft Heaven AI", page_icon="🏢")
st.title("🤖 Mysoft Heaven (BD) Ltd AI Assistant")

# Display chat history
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)


# -------------------------------------------------------------------
# User Query
# -------------------------------------------------------------------
query = st.chat_input("Ask about Mysoft Heaven...")

if query:

    # Save user message
    st.session_state.chat_history.append(("user", query))

    # Show user message
    st.chat_message("user").markdown(query)

    # Retrieve similar docs
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    # Build conversation history text manually
    history_text = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])

    # Create final prompt
    final_prompt = prompt.format(
        context=context,
        question=query,
        history=history_text
    )

    # LLM response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke(final_prompt)
                answer = response.content
            except Exception as e:
                answer = f"Error: {e}"

            st.markdown(answer)

    # Save assistant response
    st.session_state.chat_history.append(("assistant", answer))
