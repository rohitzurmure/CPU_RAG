import streamlit as st
import pandas as pd
from rag_pipeline import load_documents, chunk_documents, create_vector_store, build_rag_chain
from evaluate import run_benchmark

# Configure the Streamlit page
st.set_page_config(page_title="CPU RAG with ChatOllama", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) with ChromaDB + ChatOllama")
st.markdown("Ask a question about the source documents, or run a full evaluation to benchmark the system's performance.")

# --- Pipeline Initialization ---
@st.cache_resource
def init_pipeline():
    """
    Cached function to initialize the RAG pipeline.
    This prevents re-loading models and data on every interaction.
    """
    with st.spinner("Initializing RAG Pipeline... This may take a few moments."):
        docs = load_documents()
        chunks = chunk_documents(docs)
        vectordb = create_vector_store(chunks)
        chain = build_rag_chain(vectordb)
    return chain

chain = init_pipeline()



st.header("Ask a Question")
query = st.text_input("Enter your question based on the provided documents:", "")

if st.button("Get Answer"):
    if query.strip() and chain:
        with st.spinner("Generating response..."):
            result = chain.invoke({"query": query})
            response = result.get("result", "No response found.")
            sources = result.get("source_documents", [])

            st.markdown("# Answer")
            st.write(response)

    else:
        st.warning("Please enter a question.")



st.markdown("---")
st.header("Testing and Benchmarking")
st.write("Click the button below to run a predefined set of tests to evaluate the RAG system's accuracy, retrieval quality, and speed.")

if st.button("Run Full Evaluation"):
    if chain:
        with st.spinner("Running evaluation on the test dataset... This may take a while."):
            benchmark_df = run_benchmark(chain)

            st.markdown("#Detailed Evaluation Results")
            st.dataframe(benchmark_df)

            st.markdown("#Average Metrics")
            avg_retrieval = benchmark_df["Retrieval Score (0/1)"].mean()
            avg_accuracy = benchmark_df["Accuracy (BERTScore F1)"].mean()
            avg_latency = benchmark_df["Latency (s)"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg. Retrieval Score", f"{avg_retrieval:.2f}", help="Score of 1 indicates relevant documents were successfully retrieved.")
            col2.metric("Avg. Accuracy (BERTScore F1)", f"{avg_accuracy:.2f}", help="Semantic similarity between the generated answer and the ground truth. Higher is better.")
            col3.metric("Avg. Latency (s)", f"{avg_latency:.2f} s", help="Average time taken to generate a response.")
    else:
        st.error("The RAG chain is not initialized. Cannot run evaluation.")
