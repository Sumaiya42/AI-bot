import streamlit as st
import os
from dotenv import load_dotenv

# -------------------------
# Document Loading & Processing
# -------------------------
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq LLM
from langchain_groq import ChatGroq

# LangChain utilities
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# Load API Keys from .env
# -------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found! Please add it to your .env file.")
    st.stop()

# -------------------------
# Document Chunking & Vector DB Setup
# -------------------------
def initialize_rag():
    if not os.path.exists('data/'):
        os.makedirs('data/')
        return None

    # Load PDFs
    loader = DirectoryLoader('data/', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    if not docs:
        return None

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Local embeddings (HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Mysoft Heaven AI", page_icon="🏢")
st.title("🏢 Mysoft Heaven (BD) Ltd. AI Assistant")

vectorstore = initialize_rag()

if vectorstore is None:
    st.info("Please put Mysoft Heaven's PDF documents in the 'data' folder to start.")
else:
    # Groq LLM Configuration
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_retries=2
    )

    # Strict system prompt
    system_prompt = (
        "You are a professional assistant for Mysoft Heaven (BD) Ltd. "
        "Answer questions strictly based on provided company documents. "
        "If the answer is not in the context, say: 'I apologize, but I don't have information about that.' "
        "Do not answer questions irrelevant to the company.\n\n"
        "Context: {context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    qa_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    # -------------------------
    # Chat history
    # -------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_input := st.chat_input("Ask anything about Mysoft Heaven..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking (Powered by Groq)..."):
                try:
                    response = qa_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
