import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline

# Initialize Embedding Model
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Vector Store
def initialize_vector_store(embeddings, directory="vector_store"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return Chroma(persist_directory=directory, embedding_function=embeddings)

# Local LLM Setup
def load_local_llm():
    # Load model and tokenizer
    model_name = "gpt2"  # Replace with a better local model if available
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Configure pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # Use CPU; set GPU index if GPU is available
        max_length=1024  # Ensure longer context handling
    )

    # Wrap pipeline in LangChain's HuggingFacePipeline
    return HuggingFacePipeline(pipeline=llm_pipeline)

# Truncate Query Input to Prevent Length Issues
def truncate_input(query, tokenizer, max_length=1024):
    tokens = tokenizer.encode(query, return_tensors="pt")
    if len(tokens[0]) > max_length:
        return tokenizer.decode(tokens[0][:max_length], skip_special_tokens=True)
    return query

# PDF Processing and Adding to Vector Store
def process_pdfs(pdf_files, vector_store):
    for pdf in pdf_files:
        st.info(f"Processing: {pdf.name}")
        loader = PyPDFLoader(pdf)
        documents = loader.load()
        vector_store.add_documents(documents)
    vector_store.persist()
    st.success("Vector store updated!")

# Configure Query Engine
def configure_query_engine(vector_store, llm):
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False  # Adjust if you want source documents included
    )

# Streamlit UI
st.title("Content Engine: PDF Analysis & Comparison")
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload Form 10-K PDFs", type="pdf", accept_multiple_files=True)
query = st.text_input("Ask a question about the PDFs (e.g., 'What are the risk factors for Tesla?')")

# Load Models and Vector Store
embedding_model = load_embedding_model()
vector_store = initialize_vector_store(embedding_model)
local_llm = load_local_llm()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Handle PDF Uploads
if uploaded_files:
    st.sidebar.write(f"{len(uploaded_files)} files uploaded.")
    if st.sidebar.button("Process PDFs"):
        process_pdfs(uploaded_files, vector_store)

# Handle User Query
if query:
    query = truncate_input(query, tokenizer, max_length=1024)  # Truncate query if necessary
    st.write(f"**Your Query:** {query}")
    query_engine = configure_query_engine(vector_store, local_llm)

    # Run query directly
    response = query_engine.run(query)
    st.write(f"**Response:** {response}")
