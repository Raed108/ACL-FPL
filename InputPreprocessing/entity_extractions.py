from urllib import response
from google import genai
from google.genai import types
import os 
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
import spacy
import re

URI = os.getenv("URI")
USERNAME = os.getenv("NeoName")
PASSWORD = os.getenv("PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

from pydantic import BaseModel, Field
from typing import List, Optional


class Entity(BaseModel):
    player_name: List[str] = Field(
        default_factory=list,
        description="List of player names mentioned in the query. May be empty if no player names are detected."
    )
    team: List[str] = Field(
        default_factory=list,
        description="List of team names detected in the query. May be empty if no teams are detected."
    )
    season: List[str] = Field(
        default_factory=list,
        description="List of football seasons referenced in the query and should be full season format like \"2022-23\" e.g if user enters(2022 or 22), and we have only 2 seasons 2021-22 and 2022-23, if else return empty list"
    )
    gameweek: List[str] = Field(
        default_factory=list,
        description="List of gameweeks mentioned in the query (e.g., 'GW12'). May be empty."
    )
    position: List[str] = Field(
        default_factory=list,
        description="List of football player positions extracted from the query (e.g., 'DEF','MID'). May be empty."
    )
    statistic: List[str] = Field(
        default_factory=list,
        description="List of statistical attributes referenced in the query (e.g., 'goals', 'assists'). May be empty."
    )



def extract_entities_with_llm(user_query: str):
    prompt = f"""

    Respond with VALID JSON ONLY.

    User Query: "{user_query}"
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
        "response_mime_type": "application/json",
        "response_json_schema": Entity.model_json_schema(),
     },
    )

    json_text = json.loads(response.text)
    return json_text


###########################################


nlp = spacy.load("en_core_web_sm")







with driver.session() as session:
    result = session.run("MATCH (t:Team) WITH DISTINCT t RETURN t.name AS name")
    teams = sorted([row["name"].lower().strip() for row in result], key=len, reverse=True)

    # print(teams)


    result = session.run("MATCH (s:Season) RETURN s.season_name AS season")
    seasons = {row["season"] for row in result}




def extract_entities_spacy(text):
    doc = nlp(text)
    entities = {
        "player_name": [],
        "team": [],
        "season": [],
        "gameweek": [],
        "position": [],
        "statistic": []
    }
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["player_name"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["team"].append(ent.text)
        elif ent.label_ == "DATE":
            if ent.text.isdigit() and len(ent.text) == 4:  
                entities["season"].append(int(ent.text))
        
    return entities


def extract_gameweek(text):
    matches = re.findall(r"(?:gw|gameweek|week)\s*([0-9]+)", text.lower())
    return [int(m) for m in matches]


def extract_position(text):
    POSITION_MAP = {
        "forward": "FWD", "forwards": "FWD", "striker": "FWD", "strikers": "FWD", "fwd": "FWD", "fwds": "FWD", "attacker": "FWD", "attackers": "FWD",
        "midfielder": "MID", "midfielders": "MID", "mid": "MID", "mids": "MID", "winger": "MID", "wingers": "MID", "cm": "MID", "cmf": "MID", "cam": "MID", "cdm": "MID",
        "defender": "DEF", "defenders": "DEF", "def": "DEF", "defs": "DEF", "fullback": "DEF", "fullbacks": "DEF", "cb": "DEF", "cbf": "DEF", "lb": "DEF", "rb": "DEF",
        "goalkeeper": "GK", "keeper": "GK", "goalkeepers": "GK", "gk": "GK"
    }

    found = []
    for word, pos in POSITION_MAP.items():
        if word in text.lower():
            if pos not in found:
                found.append(pos)
    return found

def extract_team(text):
    TEAM_SYNONYMS = {
        "crystal palace": [
            "palace", "crystal", "crystal palace fc", "cpfc"
        ],

        "nott'm forest": [
            "nottingham forest", "forest", "notts forest", "nottm forest", "nottingham"
        ],

        "aston villa": [
            "villa", "aston villa fc", "avfc"
        ],

        "southampton": [
            "saints", "southampton fc", "soton"
        ],

        "bournemouth": [
            "afc bournemouth", "bournemouth fc", "cherries"
        ],

        "brentford": [
            "brentford fc", "the bees"
        ],

        "liverpool": [
            "liverpool fc", "lfc", "the reds"
        ],

        "leicester": [
            "leicester city", "leicester city fc", "lcfc", "foxes", "leicester fc"
        ],

        "newcastle": [
            "newcastle united", "newcastle utd", "newcastle united fc", "nufc", "magpies"
        ],

        "brighton": [
            "brighton & hove albion", "brighton and hove albion", "bha", "bhafc", "brighton fc", "seagulls"
        ],

        "west ham": [
            "west ham united", "west ham utd", "west ham united fc", "whu", "whufc", "hammers"
        ],

        "man city": [
            "manchester city", "man city fc", "manchester city fc", "mancity", "mcfc", "city"
        ],

        "burnley": [
            "burnley fc", "clarets"
        ],

        "norwich": [
            "norwich city", "norwich city fc", "ncfc", "canaries"
        ],

        "chelsea": [
            "chelsea fc", "cfc", "the blues"
        ],

        "everton": [
            "everton fc", "efc", "toffees"
        ],

        "watford": [
            "watford fc", "hornets"
        ],

        "man utd": [
            "manchester united", "man united", "man utd fc", "manchester utd", "manchester united fc",
            "mufc", "red devils"
        ],

        "arsenal": [
            "arsenal fc", "afc", "gunners"
        ],

        "wolves": [
            "wolverhampton wanderers", "wolverhampton", "wolves fc", "wwfc"
        ],

        "fulham": [
            "fulham fc", "ffc", "cottagers"
        ],

        "spurs": [
            "tottenham", "tottenham hotspur", "tottenham hotspur fc", "thfc"
        ],

        "leeds": [
            "leeds united", "leeds utd", "leeds united fc", "lufc"
        ]
    }

    text_l = text.lower()
    found = []

    # direct match
    for t in TEAM_SYNONYMS:
        if t in text_l and t not in found:
            found.append(t)

    # synonyms match
    for canonical, aliases in TEAM_SYNONYMS.items():
        for alias in aliases:
            if alias in text_l and canonical not in found:
                found.append(canonical)

    return found


def extract_season(text):
    found = []
    for s in seasons:
        if str(s).split("-")[0] in text:
            found.append(s)
    return found
        
def extract_statistic(text):
    found = []
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

    for stat, keywords in STATISTIC_MAP.items():
        for keyword in keywords:
            if keyword in text.lower():
                found.append(stat)
    return found


def unique_preserve_order(lst):
    # 1. Convert all items to strings for comparison
    str_values = {str(x) for x in lst}
    
    seen = set()
    result = []
    
    for item in lst:
        s_item = str(item)
        
        # Check if this item is just a prefix of another item in the list
        # e.g., if item is 2022, and "2022-23" is also in the list, skip 2022
        is_redundant = False
        for other in str_values:
            if other != s_item and other.startswith(s_item) and len(other) > len(s_item):
                is_redundant = True
                break
        
        if is_redundant:
            continue

        # Standard deduplication
        key = s_item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
            
    return result



def extract_entities(text):
    entities = extract_entities_spacy(text)

    # deterministic logic
    entities["gameweek"] += extract_gameweek(text)
    entities["position"] += extract_position(text)
    entities["team"] += extract_team(text)
    entities["season"] += extract_season(text)
    entities["statistic"] += extract_statistic(text)


    for key in entities:
        entities[key] = unique_preserve_order(entities[key])

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