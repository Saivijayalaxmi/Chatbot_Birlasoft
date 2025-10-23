# --- Step 2: Import Libraries ---
import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
# --- Step 3: Cached Backend Functions ---

@st.cache_resource(show_spinner="📘 Processing Birlasoft Annual Report...")
def create_vector_db(uploaded_file):
    """Creates a Chroma vector database from the uploaded Birlasoft Annual Report PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Load and Split PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)
        documents = [Document(page_content=doc.page_content) for doc in docs]
        
        # Create Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create Vector Database
        vector_db = Chroma.from_documents(documents, embedding=embeddings)
        return vector_db
    
    finally:
        os.unlink(tmp_path)


@st.cache_resource
def get_llm(api_key):
    """Initializes the ChatOpenAI model for Q&A."""
    return ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        temperature=0.2,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=500,
        api_key=api_key
    )


def get_response(llm, retriever, chat_history, question):
    """Generates a context-aware response using the Birlasoft Annual Report."""
    
    template = """You are an intelligent assistant for analyzing Birlasoft’s Annual Report.
Use the following retrieved context to answer accurately.
If information is not found, state that clearly.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(template)

    # Prepare chat history
    chat_history_text = "No previous conversation."
    if chat_history:
        formatted = []
        for role, msg in chat_history[-6:]:
            formatted.append(f"{role.capitalize()}: {msg}")
        chat_history_text = "\n".join(formatted)

    # Retrieve relevant chunks
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # Format prompt and get response
    formatted_prompt = prompt_template.format(
        context=context,
        chat_history=chat_history_text,
        question=question
    )

    answer = llm.invoke(formatted_prompt).content
    return answer
# --- Step 4: Streamlit Interface ---

st.set_page_config(
    page_title="💼 Birlasoft Annual Report Chatbot",
    layout="wide",
    page_icon="🤖"
)

# Session State Setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Sidebar
with st.sidebar:
    st.title("📄 Birlasoft Annual Report Q&A")
    st.markdown("Upload the **Birlasoft Annual Report (PDF)** and interact with it intelligently!")
    
    uploaded_file = st.file_uploader("📤 Upload Birlasoft Annual Report", type=["pdf"])
    
    api_key = st.text_input(
        "🔑 Enter your OpenRouter API Key",
        type="password",
        help="Get your free key from https://openrouter.ai/"
    )
    
    if api_key:
        st.success("✅ API key entered successfully!")
    else:
        st.warning("⚠️ Please enter your OpenRouter API key.")
    
    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        create_vector_db.clear()
        st.rerun()
# --- Step 5: Main Chat Interaction ---

st.title("🤖 Birlasoft Annual Report - Smart Q&A Assistant")

# Process uploaded file
if uploaded_file:
    if st.session_state.vector_db is None:
        st.session_state.vector_db = create_vector_db(uploaded_file)
        st.success("✅ Report processed successfully! Ask your questions below.")

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Chat Input
if user_question := st.chat_input("Ask a question about Birlasoft's Annual Report..."):
    if not api_key:
        st.warning("Please enter your OpenRouter API key first.")
        st.stop()
    
    if st.session_state.vector_db is None:
        st.warning("Please upload the Birlasoft Annual Report PDF first.")
        st.stop()

    # Show user message
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate bot answer
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing the report..."):
            llm = get_llm(api_key)
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
            answer = get_response(llm, retriever, st.session_state.chat_history, user_question)
            st.markdown(answer)
    
    # Store assistant message
    st.session_state.chat_history.append(("assistant", answer))

elif not st.session_state.chat_history:
    st.info("Upload the Birlasoft Annual Report PDF to start chatting!")
