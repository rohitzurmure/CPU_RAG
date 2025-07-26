import os
import shutil
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

def load_documents():
    """
    Loads a predefined set of documents.
    In a real-world application, this would be replaced with a file loader
    (e.g., PyPDFLoader, UnstructuredFileLoader) or a database connection.
    """
    # Expanded documents to provide more context for evaluation
    docs_content = [
        "The mitochondria is the powerhouse of the cell, responsible for generating most of the cell's supply of adenosine triphosphate (ATP).",
        "Paris is the capital and most populous city of France, known for its art, fashion, gastronomy and culture.",
        "LangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs). It helps build LLM apps with modular components.",
        "The Earth revolves around the Sun in an elliptical orbit, completing one revolution in approximately 365.25 days.",
        "Water is a chemical compound with the formula H2O, meaning one molecule of water contains one oxygen atom and two hydrogen atoms."
    ]
    return [Document(page_content=doc) for doc in docs_content]

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits the documents into smaller chunks for processing.
    This is important for fitting the context into the model's window.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def create_vector_store(docs, persist_directory="chroma_store"):
    """
    Creates and persists a ChromaDB vector store from the documents.
    It uses HuggingFace embeddings to convert text into numerical vectors.
    """
    # Use a standard, high-performance embedding model
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Clean up any existing vector store directory to ensure a fresh start
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # Create the vector store from documents and persist it
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

def build_rag_chain(vectordb):
    
    # Set up the retriever to fetch the top 3 most similar documents
    retriever = vectordb.as_retriever(search_type="similarity", k=3)

    llm = OllamaLLM(model="gemma3:1b")

    # Create the RetrievalQA chain
    # 'stuff': Puts all retrieved text directly into the prompt.
    # 'return_source_documents': Ensures the retrieved docs are returned with the answer.
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True # This is crucial for evaluating retrieval
    )
    return chain

