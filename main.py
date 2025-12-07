from dotenv import load_dotenv
load_dotenv()  


from InputPreprocessing.entity_extractions import extract_entities, extract_entities_with_llm
from InputPreprocessing.intent_classifier import classify_intent
from InputPreprocessing.input_embedding import embed_user_query




print(classify_intent("How many goals has Harry Kane scored this season?"))
print(extract_entities("Show me the top midfielders from Arsenal in season 2022/23 with most assists"))
embedding = embed_user_query("Who are the top forwards in the 2023 season?")
print("Embedding length:", len(embedding))
