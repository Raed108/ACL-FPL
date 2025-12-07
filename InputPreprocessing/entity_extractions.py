from google import genai
from google.genai import types
import os 
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
import spacy

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config.txt")

with open(CONFIG_PATH) as f:
    lines = [line.strip() for line in f if line.strip() and "=" in line]
    URI = [l for l in lines if l.startswith("URI=")][0].split("=", 1)[1]
    USERNAME = [l for l in lines if l.startswith("USERNAME=")][0].split("=", 1)[1]
    PASSWORD = [l for l in lines if l.startswith("PASSWORD=")][0].split("=", 1)[1]

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# load_dotenv("../.env")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def extract_entities_with_llm(user_query: str):
    prompt = f"""
    Extract the following entities from the user query.
    If an entity is missing, return null.

    Entities:
    - player_name
    - team
    - position (map to one of: GK, DEF, MID, FWD)
    - gameweek (integer)
    - season (year, e.g., 2023-24)
    - statistic (map to : goals, assists, saves, minutes, bonus, clean sheets,
      goals conceded, own goals, penalties saved, penalties missed,
      yellow cards, red cards, total points, bps, form, threat,
      creativity, influence)

    Return output as valid JSON ONLY.

    User Query: "{user_query}"
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.candidates[0].content.parts[0].text
    # return json.loads(text)
    return text

###########################################


nlp = spacy.load("en_core_web_sm")

POSITION_MAP = {
    "forward": "FWD", "forwards": "FWD", "striker": "FWD", "strikers": "FWD", "fwd": "FWD", "fwds": "FWD", "attacker": "FWD", "attackers": "FWD",
    "midfielder": "MID", "midfielders": "MID", "mid": "MID", "mids": "MID", "winger": "MID", "wingers": "MID", "cm": "MID", "cmf": "MID", "cam": "MID", "cdm": "MID",
    "defender": "DEF", "defenders": "DEF", "def": "DEF", "defs": "DEF", "fullback": "DEF", "fullbacks": "DEF", "cb": "DEF", "cbf": "DEF", "lb": "DEF", "rb": "DEF",
    "goalkeeper": "GK", "keeper": "GK", "goalkeepers": "GK", "gk": "GK"
}


with driver.session() as session:
    result = session.run("MATCH (t:Team) WITH DISTINCT t RETURN t.name AS name")
    teams = {row["name"].lower() for row in result}

    result = session.run("MATCH (s:Season) RETURN s.season_name AS season")
    seasons = {row["season"] for row in result}




def extract_entities_spacy(text):
    doc = nlp(text)
    entities = {
        "player_name": None,
        "team": None,
        "season": None,
        "gameweek": None,
        "position": None,
        "statistic": None
    }
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["player_name"] = ent.text
        elif ent.label_ == "ORG":
            entities["team"] = ent.text
        elif ent.label_ == "DATE":
            if ent.text.isdigit() and len(ent.text) == 4:  
                entities["season"] = int(ent.text)
        
    return entities

import re

def extract_gameweek(text):
    # gw5, gw 5, gameweek 5, week 5
    match = re.search(r"(?:gw|gameweek|week)\s*([0-9]+)", text.lower())
    return int(match.group(1)) if match else None

def extract_position(text):
    text = text.lower()
    for word, pos in POSITION_MAP.items():
        if word in text:
            return pos
    return None

def extract_team(text):
    for t in teams:
        if t in text.lower():
            return t
    return None

def extract_season(text):
    for s in seasons:
        if str(s).split("-")[0] in text:
            return s
        
def extract_statistic(text):
    STATISTIC_MAP = {
        "goals": ["goal", "goals", "scored"],
        "assists": ["assist", "assists"],
        "saves": ["save", "saves"],
        "minutes": ["minute", "minutes", "played"],
        "bonus": ["bonus", "bonuses"],
        "clean sheets": ["clean sheet", "clean sheets"],
        "goals conceded": ["goal conceded", "goals conceded", "conceded"],
        "own goals": ["own goal", "own goals"],
        "penalties saved": ["penalty saved", "penalties saved"],
        "penalties missed": ["penalty missed", "penalties missed"],
        "yellow cards": ["yellow card", "yellow cards"],
        "red cards": ["red card", "red cards"],
        "total points": ["total points", "points"],
        "bps": ["bps", "bonus points system"],
        "form": ["form"],
        "threat": ["threat"],
        "creativity": ["creativity"],
        "influence": ["influence"]
    }

    text = text.lower()
    for stat, keywords in STATISTIC_MAP.items():
        for keyword in keywords:
            if keyword in text:
                return stat
    return None



def extract_entities(text):
    entities = extract_entities_spacy(text)

    # deterministic logic
    entities["gameweek"] = entities["gameweek"] or extract_gameweek(text)
    entities["position"] = entities["position"] or extract_position(text)
    entities["team"] = entities["team"] or extract_team(text)
    entities["season"] = entities["season"] or extract_season(text)
    entities["statistic"] = entities["statistic"] or extract_statistic(text)


    return entities


# print("first example without llm:")
# print(extract_entities("Show me the top midfielders from Arsenal in season 2022/23 with most assists"))
# print("first example with llm:")
# print(extract_entities_with_llm("Show me the top midfielders from Arsenal in season 2022/23 with most assists"))


# print("second example without llm:")
# print(extract_entities("How many goals did Harry Kane score in gameweek 25 of season 2022?"))
# print("second example with llm:")
# print(extract_entities_with_llm("How many goals did Harry Kane score in gameweek 25 of season 2022?"))

# print("third example without llm:")
# print(extract_entities("Who are the defenders with the highest clean sheets in season 2021?"))
# print("third example with llm:")
# print(extract_entities_with_llm("Who are the defenders with the highest clean sheets in season 2021?"))

# print("fourth example without llm:")
# print(extract_entities("Who is Mohamed Salah's next fixture for Liverpool?"))
# print("fourth example with llm:")
# print(extract_entities_with_llm("Who is Mohamed Salah's next fixture for Liverpool?"))