import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY in Streamlit Secrets")
    st.stop()


# Load PDF & build vectorstore
def load_vectorstore():
    if not os.path.exists("data"):
        return None
    loader = PyPDFLoader("Data/Soil Classification.pdf") 
    docs = loader.load()


    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


vectorstore = load_vectorstore()
if vectorstore is None:
    st.error("No PDF found in data folder")
    st.stop()


# LLM setup
llm = ChatGroq(model="llama-3.3-70b-versatile")


# Prompt template
system_prompt = """
You are My AI assistant.
Answer ONLY from company documents.

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])


# Streamlit UI
st.set_page_config(page_title="Soil classification AI", page_icon="🏢")
st.title("🤖 Soil classification AI Assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
query = st.chat_input("Ask about My Soil classification...")

if query:
    # Show user message instantly
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Similarity search
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([d.page_content for d in docs])
    final_prompt = prompt.format(context=context, question=query)

    # LLM response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke(final_prompt)
                answer = response.content
            except Exception as e:
                answer = f"Error: {e}"

            st.markdown(answer)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})
   












