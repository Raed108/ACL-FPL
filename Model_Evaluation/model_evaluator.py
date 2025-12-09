import time
import requests
import pandas as pd

# ------------------------
# 1) Define LLMs to test
# ------------------------
MODELS = {
    "llama3": "meta-llama/llama-3-8b-instruct",
    "gemma2": "google/gemma-2-9b-it",
    "qwen2": "qwen/qwen2-7b-instruct"
}

# Example test queries
TEST_QUERIES = [
    {"question": "How many total players are in the dataset?", "expected_keywords": ["players"]},
    {"question": "Which player scored the most total points in season 2022-23?", "expected_keywords": ["player", "points"]},
    {"question": "Who are the top midfielders from Arsenal in season 2022/23 with most assists?", "expected_keywords": ["midfielders", "Arsenal", "assists"]}
]

# ------------------------
# 2) Query LLM
# ------------------------
def query_llm(model_name, kg_context, user_question, api_key):
    """
    Sends structured prompt to LLM and returns answer + metrics.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"  # example
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    prompt = f"""
Context (from KG):
{kg_context}

Persona:
You are an FPL expert assistant that answers fantasy football questions using only the KG context.

Task:
Answer the following user question using only the information above. Do not hallucinate.

User Question:
"{user_question}"
    """
    
    body = {"model": model_name, "messages":[{"role":"user","content":prompt}]}

    start_time = time.time()
    response = requests.post(url, headers=headers, json=body)
    latency = time.time() - start_time

    try:
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
    except:
        answer = "Error: No output from model"
        tokens_in = tokens_out = total_tokens = 0

    return {
        "answer": answer,
        "response_time": latency,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": total_tokens
    }

# ------------------------
# 3) Accuracy metric
# ------------------------
def compute_accuracy(answer, expected_keywords):
    if not expected_keywords: return None
    correct = sum(1 for k in expected_keywords if k.lower() in answer.lower())
    return correct / len(expected_keywords)

# ------------------------
# 4) Qualitative metrics heuristics
# ------------------------
def compute_qualitative_metrics(answer, expected_keywords, kg_context):
    # Relevance: fraction of KG keywords found in answer
    kg_keywords = set(word.lower() for word in kg_context.split())
    answer_words = set(word.lower() for word in answer.split())
    qualitative_relevance = len(kg_keywords & answer_words) / max(1, len(kg_keywords))

    # Correctness: same as accuracy
    qualitative_correctness = compute_accuracy(answer, expected_keywords)

    # Completeness: 1 if all expected keywords present, else fraction
    qualitative_completeness = sum(1 for k in expected_keywords if k.lower() in answer.lower()) / max(1, len(expected_keywords))

    # Naturalness: 1 if non-empty answer, 0 otherwise
    qualitative_naturalness = 1.0 if answer.strip() and "Error" not in answer else 0.0

    # Confidence: 0 if "I can't" or "Error" in answer, else 1
    qualitative_confidence = 0.0 if "I can't" in answer or "Error" in answer else 1.0

    return qualitative_relevance, qualitative_correctness, qualitative_completeness, qualitative_naturalness, qualitative_confidence

# ------------------------
# 5) Run evaluation across models
# ------------------------
def evaluate_models(kg_context, api_key):
    results = []

    for model_name, model_id in MODELS.items():
        print(f"\nEvaluating model: {model_name}")
        for test in TEST_QUERIES:
            q = test["question"]
            expected = test["expected_keywords"]

            output = query_llm(model_id, kg_context, q, api_key)

            # Compute metrics
            accuracy = compute_accuracy(output["answer"], expected)
            relevance, correctness, completeness, naturalness, confidence = compute_qualitative_metrics(output["answer"], expected, kg_context)

            results.append({
                "model": model_name,
                "question": q,
                "answer": output["answer"],
                "accuracy": accuracy,
                "response_time": output["response_time"],
                "tokens_in": output["tokens_in"],
                "tokens_out": output["tokens_out"],
                "total_tokens": output["total_tokens"],
                "qualitative_relevance": relevance,
                "qualitative_correctness": correctness,
                "qualitative_completeness": completeness,
                "qualitative_naturalness": naturalness,
                "qualitative_confidence": confidence
            })

    df = pd.DataFrame(results)
    df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved results → model_comparison_results.csv")

# ------------------------
# 6) Example usage
# ------------------------
if __name__ == "__main__":
    # Example KG context string (replace with your actual KG query result)
    example_kg_context = "- Player: Harry Kane, Total Points: 210\n- Player: Mohamed Salah, Total Points: 205\n"
    OPENROUTER_API_KEY = "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"
    evaluate_models(example_kg_context, OPENROUTER_API_KEY)
