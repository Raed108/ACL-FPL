import time
import requests
import pandas as pd

MODELS = {
    "llama3": "meta-llama/llama-3-8b-instruct",
    "gemma2": "google/gemma-2-9b-it",
    "mistralai": "mistralai/mistral-7b-instruct:free"
}

def query_llm(model_name, prompt_str, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model_name, "messages":[{"role":"user","content":prompt_str}]}

    start_time = time.time()
    response = requests.post(url, headers=headers, json=body)
    latency = time.time() - start_time

    try:
        data = response.json()
        if "choices" in data:
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
        elif "result" in data:  
            answer = data["result"]
            usage = {}
        else:
            answer = f"Error: unexpected response format {data}"
            usage = {}

        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

    except Exception as e:
        answer = f"Error: {str(e)}"
        tokens_in = tokens_out = total_tokens = 0

    return {
        "answer": answer,
        "response_time": latency,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": total_tokens
    }


def compute_accuracy(answer, expected_keywords):
    if not expected_keywords: 
        return None
    correct = sum(1 for k in expected_keywords if k.lower() in answer.lower())
    return correct / len(expected_keywords)


class ModelEvaluator:
    def __init__(self, prompt_template, api_key, test_queries_file="test_queries.txt"):
        self.prompt_template = prompt_template
        self.api_key = api_key
        self.test_queries_file = test_queries_file
        self.test_queries = self._load_test_queries()

    def _load_test_queries(self):
        with open(self.test_queries_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def evaluate(self):
        results = []

        for model_name, model_id in MODELS.items():
            print(f"\nEvaluating model: {model_name}")
            for q in self.test_queries:
                expected_keywords = [w for w in q.lower().split() if len(w) > 3]

                prompt_str = self.prompt_template.format(question=q)
                output = query_llm(model_id, prompt_str, self.api_key)

                accuracy = compute_accuracy(output["answer"], expected_keywords)

                # qualitative fields left EMPTY for human input
                results.append({
                    "model": model_name,
                    "question": q,
                    "answer": output["answer"],
                    "accuracy": accuracy,
                    "response_time": output["response_time"],
                    "tokens_in": output["tokens_in"],
                    "tokens_out": output["tokens_out"],
                    "total_tokens": output["total_tokens"],
                    "qualitative_relevance": "",
                    "qualitative_correctness": "",
                    "qualitative_completeness": "",
                    "qualitative_naturalness": "",
                    "qualitative_confidence": ""
                })

        df = pd.DataFrame(results)
        df.to_csv("model_comparison_results.csv", index=False)
        print("\nSaved results → model_comparison_results.csv")

        # Best model = only quantitative (accuracy)
        summary = df.groupby("model")[["accuracy"]].mean()
        best_model = summary["accuracy"].idxmax()

        print("\nBest model:", best_model)
        print(summary)
        return df, best_model


# ==================== Example usage ====================
if __name__ == "__main__":
    prompt_template = """
Context:
- Player: Harry Kane, Total Points: 210
- Player: Mohamed Salah, Total Points: 205
- Player: Arsenal Top Midfielder: Kevin De Bruyne, Assists: 12

Persona:
You are an FPL expert assistant.

Task:
Answer the user's question using only the information above.

Question:
{question}
"""

    OPENROUTER_API_KEY = "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"

    evaluator = ModelEvaluator(prompt_template, OPENROUTER_API_KEY)
    df_results, best_model = evaluator.evaluate()
