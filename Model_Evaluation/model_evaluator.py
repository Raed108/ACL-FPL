import time
import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

load_dotenv()

from LLMLayer.Baseline_Embeddings_Combined import combine_retrieval_results
from LLMLayer.Prompt_Structure import create_prompt_template
from InputPreprocessing.intent_classifier import classify_intent_llm
from InputPreprocessing.entity_extractions import extract_entities_with_llm
from GraphRetrievalLayer.embedding import answer_query



OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODELS = {
    "llama3.3": "meta-llama/llama-3.3-70b-instruct:free",
    "gemma3": "google/gemma-3-12b-it:free",
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
        "total_tokens": total_tokens,
        "model" : model_name
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
    
    def build_prompt(self, question, combined_context):
        return f"""
    Context:
    {combined_context}

    Persona:
    You are an expert assistant.

    Task:
    Answer the user's question using ONLY the context above.

    Question:
    {question}
"""


    def evaluate(self, embed_model):
        results = []

        for model_name, model_id in MODELS.items():
            print(f"\nEvaluating model: {model_name}")

            for q in self.test_queries:
                expected_keywords = [w for w in q.lower().split() if len(w) > 3]

                # ===== Retrieval pipeline =====
                intent = classify_intent_llm(q)
                entities = extract_entities_with_llm(q)

                vector_results = answer_query(
                    q,
                    entities=entities,
                    intent=intent,
                    model_choice=embed_model
                )

                combined_context = combine_retrieval_results(
                    hybrid_results=vector_results
                )

                # ===== Dynamic prompt =====
                prompt_str = self.build_prompt(q, combined_context)

                # ===== Query model =====
                output = query_llm(model_id, prompt_str, self.api_key)

                accuracy = compute_accuracy(output["answer"], expected_keywords)

                results.append({
                    "model": model_name,
                    "question": q,
                    "answer": output["answer"],
                    "accuracy": accuracy,
                    "response_time": output["response_time"],
                    "tokens_in": output["tokens_in"],
                    "tokens_out": output["tokens_out"],
                    "total_tokens": output["total_tokens"],
                    "cost": "free",
                    "qualitative_relevance": "",
                    "qualitative_correctness": "",
                    "qualitative_completeness": "",
                    "qualitative_naturalness": "",
                    "qualitative_confidence": ""
                })

        df = pd.DataFrame(results)
        df.to_csv("model_comparison_results.csv", index=False)

        summary = df.groupby("model")[["accuracy"]].mean()
        best_model = summary["accuracy"].idxmax()

        print("\nBest model:", best_model)
        print(summary)

        return df, best_model



if __name__ == "__main__":
    embed_model = "minilm"

    evaluator = ModelEvaluator(
        prompt_template=None,  # no longer needed
        api_key=OPENROUTER_API_KEY
    )

    df_results, best_model = evaluator.evaluate(embed_model)

#     prompt_template = """
# Context:
# - Player: Harry Kane, Total Points: 210
# - Player: Mohamed Salah, Total Points: 205
# - Player: Arsenal Top Midfielder: Kevin De Bruyne, Assists: 12

# Persona:
# You are an FPL expert assistant.

# Task:
# Answer the user's question using only the information above.

# Question:
# {question}
# """


#     evaluator = ModelEvaluator(prompt_template, OPENROUTER_API_KEY)
#     df_results, best_model = evaluator.evaluate()

# print(query_llm("meta-llama/llama-3.3-70b-instruct:free", "what is today's date?", OPENROUTER_API_KEY))
# print(query_llm("google/gemma-3-12b-it:free", "how can i know day from night?", OPENROUTER_API_KEY))
# print(query_llm("mistralai/mistral-7b-instruct:free", "what is today's date?", OPENROUTER_API_KEY))