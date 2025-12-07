from neo4j import GraphDatabase
import pandas as pd

with open("config.txt") as f:
    lines = [line.strip() for line in f if line.strip() and "=" in line]
    URI = [l for l in lines if l.startswith("URI=")][0].split("=", 1)[1]
    USERNAME = [l for l in lines if l.startswith("USERNAME=")][0].split("=", 1)[1]
    PASSWORD = [l for l in lines if l.startswith("PASSWORD=")][0].split("=", 1)[1]

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

df = pd.read_csv("fpl_two_seasons.csv")

query = """
UNWIND $rows AS row
MERGE (s:Season { season_name: row.season})
MERGE (gw:Gameweek { season: row.season, GW_number: toInteger(row.GW)})
MERGE (f:Fixture { season: row.season, fixture_number: toInteger(row.fixture)})
SET f.kickoff_time = row.kickoff_time
MERGE (ht:Team { name: row.home_team})
MERGE (at:Team { name: row.away_team})
MERGE (pl:Player { player_name: row.name, player_element: row.element})
MERGE (po:Position { name: row.position}) 

MERGE (s) -[:HAS_GW]->(gw)
MERGE (gw) -[:HAS_FIXTURE]->(f)
MERGE (f) -[:HAS_HOME_TEAM]->(ht)
MERGE (f) -[:HAS_AWAY_TEAM]->(at)
MERGE (pl) -[:PLAYS_AS]->(po)
MERGE (pl) -[r:PLAYED_IN]-> (f)
SET r.minutes = toInteger(row.minutes),
    r.goals_scored = toInteger(row.goals_scored),
    r.assists = toInteger(row.assists),
    r.total_points = toInteger(row.total_points),
    r.bonus = toInteger(row.bonus),
    r.clean_sheets = toInteger(row.clean_sheets),
    r.goals_conceded = toInteger(row.goals_conceded),
    r.own_goals = toInteger(row.own_goals),
    r.penalties_saved = toInteger(row.penalties_saved),
    r.penalties_missed = toInteger(row.penalties_missed),
    r.yellow_cards = toInteger(row.yellow_cards),
    r.red_cards = toInteger(row.red_cards),
    r.saves = toInteger(row.saves),
    r.bps = toInteger(row.bps),
    r.influence = toFloat(row.influence),
    r.creativity = toFloat(row.creativity),
    r.threat = toFloat(row.threat),
    r.ict_index = toFloat(row.ict_index),
    r.form = toFloat(row.form)
"""
batch_size = 1000
total_rows = len(df)

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

with driver.session() as session:
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size].to_dict('records')
        session.run(query, rows=batch)
        print(f"Processed rows {i} to {min(i+batch_size, total_rows)}")


driver.close()













