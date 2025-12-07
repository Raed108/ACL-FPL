# from neo4j import GraphDatabase
from google import genai
from google.genai import types
import os 


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def classify_intent_llm(user_input: str):
    prompt = f"""
    Classify the user's query into ONE intent from the following list:
    - player_stats
    - top_players
    - fixture_query
    - team_analysis
    - recommendation

    Only respond with the intent name.

    User Query: "{user_input}"
    """

    chat = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt,
        # config= types.GenerateContentConfig(system_instruction=prompt)
    )

    intent = chat.candidates[0].content.parts[0].text

    return intent


def classify_intent(user_input: str) -> str:
    """
    Classifies FPL user queries into one of 6 intents using keyword matching.
    Fast, deterministic, and fallback to LLM if no keywords match.
    """
    text = user_input.lower().strip()

    # 1. Recommendation / Best / Top / Suggest
    if any(word in text for word in ["recommend", "suggest", "best", "top", "captain", "vice-captain",
                                     "who should i pick", "who to buy", "who to transfer", "budget",
                                     "who should i pick", "who to buy", "who to transfer", "captain", "vice-captain"]):
        return "recommendation"

    # 2. Player-specific stats/performance
    if any(word in text for word in ["how many", "how much", "points", "goals", "assists", "clean sheets",
                                     "bonus", "bps", "form", "ict", "threat", "creativity", "influence",
                                     "played", "minutes", "xg", "xa", "performance", "stats", "scored"]):
        return "player_stats"

    # 3. Fixture / Match / Schedule
    if any(word in text for word in ["fixture", "match", "game", "against", "vs", "play", "when", "next game",
                                     "double gameweek", "blank gameweek", "dgw", "bgw", "schedule"]):
        return "fixture_query"

    # 4. Team / Club analysis
    if any(word in text for word in ["team", "squad", "brighton", "arsenal", "city", "united", "liverpool",
                                     "chelsea", "spurs", "newcastle", "villa", "west ham", "leicester"]):
        if "fixture" not in text and "match" not in text:
            return "team_analysis"

    # 5. Top Players / Comparison / History
    if any(word in text for word in ["highest", "top ", "best", "most", "least", "ever", "all time",
                                     "leaders", "better than", "who has the most", "highest scoring"]):
        return "top_players"

    # 6. LLM fallback
    return classify_intent_llm(user_input)

# print(classify_intent("How many goals has Harry Kane scored this season?"))
# print(classify_intent("Show me the top midfielders from Arsenal in season 2022/23 with most assists"))
# print(classify_intent("Who are the best defenders in Manchester United for gameweek 5?"))
# print(classify_intent("Which team has the toughest fixtures coming up?"))
# print(classify_intent("Can you recommend some budget forwards for my team?"))