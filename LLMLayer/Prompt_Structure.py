from langchain.prompts import PromptTemplate
from Baseline_Embeddings_Combined import combine_retrieval_results

def create_prompt_template(baseline_results, vector_results):
    """
    Creates a prompt template that combines results from baseline and vector searches.
    """
    combined_context = combine_retrieval_results(baseline_results, vector_results)
    
    # Format the combined context into a string for the prompt
    context_str = "\n".join([f"- {item}" for item in combined_context])
    
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=f"""
You are an FPL expert. Use the following context to answer the question.
Context:
{context_str}

Question: {{question}}
"""
    )
    return prompt_template