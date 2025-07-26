import time
import pandas as pd
from bert_score import score

# This is the ground truth dataset for evaluation.
# It contains queries and their expected, ideal answers.
EVAL_DATASET = [
    {
        "query": "What is the primary function of mitochondria?",
        "ground_truth": "The mitochondria is the powerhouse of the cell, responsible for generating ATP."
    },
    {
        "query": "What city is the capital of France?",
        "ground_truth": "Paris is the capital of France."
    },
    {
        "query": "What is the purpose of the LangChain framework?",
        "ground_truth": "LangChain is a framework for building applications powered by large language models."
    },
    {
        # This query tests the model's ability to handle questions where context is absent.
        "query": "What is the currency of Japan?",
        "ground_truth": "The provided documents do not contain information about the currency of Japan."
    }
]

def evaluate_retrieval(retrieved_docs, ground_truth):
    """
    Evaluates the quality of the retrieval step.
    A simple metric: returns 1 if any retrieved document contains a keyword
    from the ground truth, otherwise 0.
    """
    # A simple keyword-based check. More advanced methods could use semantic similarity.
    keywords = ground_truth.lower().split()
    for doc in retrieved_docs:
        if any(keyword in doc.page_content.lower() for keyword in keywords if len(keyword) > 3):
            return 1
    return 0

def evaluate_accuracy(generated_answer, ground_truth):
    """
    Evaluates the accuracy of the generated answer against the ground truth
    using BERTScore, which measures semantic similarity.
    Returns the F1 score.
    """
    try:
        # The 'score' function returns Precision, Recall, and F1 tensors.
        _, _, F1 = score([generated_answer], [ground_truth], lang="en", verbose=False)
        return F1.item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return 0.0 # Return a neutral score in case of an error

def run_benchmark(chain):
    """
    Runs the full benchmark suite over the evaluation dataset.
    Calculates latency, retrieval quality, and response accuracy for each query.
    """
    results = []
    for item in EVAL_DATASET:
        query = item["query"]
        ground_truth = item["ground_truth"]

        start_time = time.time()
        # The chain is invoked to get the response and source documents
        chain_result = chain.invoke({"query": query})
        end_time = time.time()

        latency = end_time - start_time
        generated_answer = chain_result["result"].strip()
        retrieved_docs = chain_result["source_documents"]

        # Calculate evaluation metrics
        retrieval_score = evaluate_retrieval(retrieved_docs, ground_truth)
        accuracy_score = evaluate_accuracy(generated_answer, ground_truth)

        results.append({
            "Query": query,
            "Ground Truth": ground_truth,
            "Generated Answer": generated_answer,
            "Retrieval Score (0/1)": retrieval_score,
            "Accuracy (BERTScore F1)": accuracy_score,
            "Latency (s)": latency
        })

    # Convert results to a pandas DataFrame for easier analysis
    return pd.DataFrame(results)

