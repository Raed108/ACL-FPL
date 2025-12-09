from langchain.prompts import PromptTemplate
from Baseline_Embeddings_Combined import combine_retrieval_results

def create_prompt_template(baseline_results, vector_results, user_question):
    """
    Creates a prompt template that combines results from baseline and vector searches.
    """
    combined_context = combine_retrieval_results(baseline_results, vector_results)
    persona = "You are an FPL expert."
    # Format the combined context into a string for the prompt
    context_str = "\n".join([f"- {item}" for item in combined_context])
    
    prompt_template = PromptTemplate(
        input_variables=[user_question],
        template=f"""
{persona} Use the following context to answer the question.
Context:
{context_str}

Question: {{{user_question}}}
"""
    )
    return prompt_template