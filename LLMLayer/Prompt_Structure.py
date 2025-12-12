from langchain.prompts import PromptTemplate

def create_prompt_template(context, user_question):
    """
    Creates and formats a prompt string using baseline and vector search results.
    """
    persona = "FPL expert"
    
    # 1. Define the template string
    template = (
        "You are an {persona}. Use the following context to answer the question.\n"
        "Context:\n{context}\n"
        "Question: {user_question}"
    )

    # 2. Create the PromptTemplate object
    prompt_template = PromptTemplate(
        template=template, 
        input_variables=["persona", "context", "user_question"]
    )
    
    # 3. Format the prompt using the function arguments and the local persona variable
    return prompt_template.format(
        persona=persona, 
        context=context, 
        user_question=user_question
    )