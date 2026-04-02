import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.memory import ConversationBufferMemory


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
# Conversation Memory (NEW)
# -------------------------------------------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True
    )
memory = st.session_state.memory


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

# Chat history display
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------------------------------------------------
# User Query
# -------------------------------------------------------------------
query = st.chat_input("Ask about Mysoft Heaven...")

if query:

    # Display user message
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve similar docs
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    # Load history from memory
    history = memory.load_memory_variables({})["history"]

    # Create final formatted prompt
    final_prompt = prompt.format(
        context=context,
        question=query,
        history=history
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

    # Save to UI history
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save to conversation memory
    memory.save_context(
        {"input": query},
        {"output": answer}
    )
