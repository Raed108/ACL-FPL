from dotenv import load_dotenv
load_dotenv()  


from InputPreprocessing.entity_extractions import extract_entities, extract_entities_with_llm
from InputPreprocessing.intent_classifier import classify_intent
from InputPreprocessing.input_embedding import embed_user_query
from GraphRetrievalLayer.Baseline import GraphRetrieval




# print(classify_intent("How many goals has Harry Kane scored this season?"))
# print(extract_entities("Show me the top midfielders and defenders from Liverpool and Arsenal in season 2022/23 with most assists"))
# embedding = embed_user_query("Who are the top forwards in the 2023 season?")
# print("Embedding length:", len(embedding))

# print(retrieve_kg_context("How many goals did Salah score in the 2022 season?"))

user_query = "How many goals did Halland score in the 2022 season?"
entities = extract_entities(user_query)
print(entities)
intent = classify_intent(user_query)
print(intent)
# graph = GraphRetrieval()
# kg_context = graph.retrieve_kg_context(entities, intent)
# print(kg_context)
# graph.close()