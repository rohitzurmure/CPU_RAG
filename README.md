 
# 🧠 CPU-Based RAG with Ollama, ChromaDB, and Streamlit

This project presents a **fully self-contained, CPU-based Retrieval-Augmented Generation (RAG) application**. It leverages open-source tools to build a chatbot capable of answering questions based on a local set of documents. The entire pipeline, from document embedding to language model inference, runs efficiently on your local machine.

The application features an interactive **Streamlit** web interface, allowing users to both ask questions and run a comprehensive evaluation suite to benchmark the system's performance on key metrics like accuracy, retrieval quality, and latency.

-----

## ✨ Features

  * **Interactive Q\&A**: A user-friendly chat interface to ask questions and receive answers from the RAG pipeline.
  * **Local First**: The entire system operates on a local CPU, eliminating the need for API keys or cloud services.
  * **Open-Source Stack**: Built exclusively with powerful open-source libraries such as **LangChain**, **Ollama**, and **ChromaDB**.
  * **Source Document Display**: Shows the exact document chunks retrieved from the vector store that were used to generate the answer.
  * **Built-in Benchmarking**: A one-click evaluation module to test the RAG system's performance.
  * **Comprehensive Metrics**: The evaluation suite measures:
      * **Response Accuracy**: Uses BERTScore to calculate the semantic similarity between the generated answer and a ground-truth reference.
      * **Retrieval Quality**: A simple but effective check to see if relevant documents were fetched.
      * **Inference Latency**: Measures the time taken to generate a response.

-----

## 🛠️ Tech Stack & Dependencies

  * **Application Framework**: Streamlit
  * **LLM Orchestration**: LangChain
  * **Local LLM Provider**: Ollama (configured with `gemma3:1b`)
  * **Vector Database**: ChromaDB
  * **Embedding Model**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
  * **Evaluation**: BERT-Score, Pandas

-----

## ⚙️ Setup and Installation

Follow these steps to get the application running on your local machine.

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2\. Install Dependencies

It's highly recommended to create a Python virtual environment first.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### 3\. Install and Set Up Ollama

If you don't have Ollama installed, follow the instructions on the [official Ollama website](https://ollama.com/download).
Once Ollama is installed, pull the `gemma:2b` model, which is used in this application for its balance of performance and resource efficiency on CPUs.

```bash
ollama pull gemma3:1b
```

-----

## 🚀 How to Run the Application

### 1\. Start the Ollama Service

Ensure the Ollama application or its background service is running before you start the Streamlit app.

### 2\. Run the Streamlit App

Open your terminal in the project's root directory and run the following command:

```bash
streamlit run app.py
```

Your web browser should automatically open to the application's URL (usually `http://localhost:8501`).

### 3\. Using the App

The application has two main functions:

  * **💬 Ask a Question**: Type a question into the text input box and click "Get Answer". The system will retrieve relevant context from the source documents and generate a response. The answer and the source documents used will be displayed.
  * **📊 Run Full Evaluation**: Click the "Run Full Evaluation" button to trigger the benchmarking process. The system will run a series of predefined questions from `evaluate.py` and display a detailed table of results, along with average scores for accuracy, retrieval, and latency.