from neo4j import GraphDatabase
import os

# Neo4j connection
URI = os.getenv("URI")
USERNAME = os.getenv("NeoName")
PASSWORD = os.getenv("PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

class GraphRetrieval:

    def __init__(self):
        self.driver = driver

    # def close(self):
    #     self.driver.close()

    #--------------------------------------------
    # Helper: run query and return results
    #--------------------------------------------
    def _run_query(self, cypher, params=None):
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(r) for r in result]

    #---------------------------------------------
    # Retrieve KG context with entities & intent
    #---------------------------------------------
    def retrieve_kg_context(self, entities, intent):
        """
        entities: dict with keys
        - player_name: list[str]
        - team: list[str]
        - position: list[str]
        - gameweek: list[int]
        - season: list[str]
        - statistic: list[str]
        intent: str, returned by classify_intent()
        """
        queries = []

       # -------------------------------------
        # Intent: Player Stats
        # -------------------------------------
        if intent == "player_stats":
            # 1. Detailed Season Overview
            queries.append("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
                WHERE ($player_name IS NULL OR toLower(p.player_name) CONTAINS toLower($player_name))
                  AND ($season IS NULL OR toLower(s.season_name) CONTAINS toLower($season))
                RETURN p.player_name AS player, 
                       s.season_name AS season,
                       sum(r.minutes) AS minutes,
                       sum(r.goals_scored) AS goals, 
                       sum(r.assists) AS assists, 
                       sum(r.clean_sheets) AS clean_sheets,
                       sum(r.total_points) AS total_points,
                       sum(r.bonus) AS total_bonus
            """)

            # 2. Recent Form (Last 5 Games Played)
            queries.append("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
                WHERE ($player_name IS NULL OR toLower(p.player_name) CONTAINS toLower($player_name))
                  AND ($season IS NULL OR toLower(s.season_name) CONTAINS toLower($season))
                WITH p, r, gw ORDER BY gw.GW_number DESC LIMIT 5
                RETURN p.player_name AS player, 
                       collect(gw.GW_number) as recent_gameweeks,
                       collect(r.total_points) as recent_points,
                       avg(r.ict_index) as avg_ict_form
            """)

            # 3. Efficiency (Points per 90)
            queries.append("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                WHERE ($player_name IS NULL OR toLower(p.player_name) CONTAINS toLower($player_name))
                WITH p, sum(r.total_points) as pts, sum(r.minutes) as mins
                WHERE mins > 0
                RETURN p.player_name AS player, 
                       (toFloat(pts) / mins * 90) AS points_per_90
            """)

        # -------------------------------------
        # Intent: Top Players
        # -------------------------------------
        elif intent == "top_players":
            # 1. Top Point Scorers
            queries.append("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (p)-[:PLAYS_AS]->(pos:Position)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
                WHERE ($position IS NULL OR toLower(pos.name) CONTAINS toLower($position))
                  AND ($season IS NULL OR toLower(s.season_name) CONTAINS toLower($season))
                RETURN p.player_name AS player, pos.name AS position, sum(r.total_points) AS total_points
                ORDER BY total_points DESC
                LIMIT 10
            """)

            # 2. Golden Boot (Goals)
            queries.append("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
                WHERE ($season IS NULL OR toLower(s.season_name) CONTAINS toLower($season))
                RETURN p.player_name AS player, sum(r.goals_scored) AS goals
                ORDER BY goals DESC
                LIMIT 5
            """)

            # 3. Top Playmakers (Assists)
            queries.append("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
                WHERE ($season IS NULL OR toLower(s.season_name) CONTAINS toLower($season))
                RETURN p.player_name AS player, sum(r.assists) AS assists, sum(r.ict_index) as creativity_score
                ORDER BY assists DESC
                LIMIT 5
            """)

            # 4. Top Defenders
            queries.append("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (p)-[:PLAYS_AS]->(pos:Position)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
                WHERE pos.name IN ['DEF', 'GK']
                  AND ($season IS NULL OR toLower(s.season_name) CONTAINS toLower($season))
                RETURN p.player_name AS player, 
                       sum(r.clean_sheets) AS clean_sheets, 
                       sum(r.goals_conceded) as goals_conceded,
                       sum(r.total_points) as total_points
                ORDER BY clean_sheets DESC, total_points DESC
                LIMIT 5
            """)

        # -------------------------------------
        # Intent: Fixture Query
        # -------------------------------------
        elif intent == "fixture_query":
            # 1. Upcoming Fixtures for Team
            queries.append("""
                MATCH (t:Team)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
                WHERE ($team IS NOT NULL AND toLower(t.name) CONTAINS toLower($team))
                  AND f.kickoff_time >= datetime() 
                WITH f, t
                MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(opponent:Team)
                WHERE opponent <> t
                RETURN f.kickoff_time AS kickoff, 
                       t.name AS team, 
                       opponent.name AS opponent
                ORDER BY f.kickoff_time ASC
                LIMIT 3
            """)

        # -------------------------------------
        # Intent: Team Analysis
        # -------------------------------------
        elif intent == "team_analysis":
            
            # 1. Best Attackers (Goals/Assists in games involving this team)
            queries.append("""
                MATCH (t:Team)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
                MATCH (p:Player)-[r:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_AS]->(pos:Position)
                WHERE ($team IS NOT NULL AND toLower(t.name) CONTAINS toLower($team))
                    AND pos.name IN ['FW', 'MID']
                RETURN p.player_name AS player, 
                       sum(r.goals_scored) AS goals, 
                       sum(r.assists) AS assists, 
                       sum(r.total_points) as points
                ORDER BY points DESC
                LIMIT 5
            """)

            # 2. Team Defensive Overview (via Clean Sheets in team games)
            queries.append("""
                MATCH (t:Team)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
                MATCH (p:Player)-[r:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_AS]->(pos:Position)
                WHERE ($team IS NOT NULL AND toLower(t.name) CONTAINS toLower($team))
                  AND pos.name IN ['DEF', 'GK']
                RETURN t.name AS team, 
                       sum(r.clean_sheets) AS total_clean_sheets, 
                       sum(r.goals_conceded) AS total_goals_conceded
            """)

            # 3. best players by total points
            queries.append("""
                MATCH (t:Team)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
                MATCH (p:Player)-[r:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_AS]->(pos:Position)
                WHERE ($team IS NOT NULL AND toLower(t.name) CONTAINS toLower($team))
                RETURN p.player_name AS player, 
                       s.season_name AS season,
                       sum(r.minutes) AS minutes,
                       sum(r.goals_scored) AS goals, 
                       sum(r.assists) AS assists, 
                       sum(r.clean_sheets) AS clean_sheets,
                       sum(r.total_points) AS total_points,
                       sum(r.bonus) AS total_bonus
                ORDER BY total_points DESC
                LIMIT 5
            """)

            # 4. Overall Team Performance (Aggregated Stats)
            queries.append("""
                MATCH (t:Team)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
                MATCH (p:Player)-[r:PLAYED_IN]->(f)
                WHERE ($team IS NOT NULL AND toLower(t.name) CONTAINS toLower($team))
                  AND ($season IS NULL OR toLower(s.season_name) CONTAINS toLower($season))
                
                RETURN t.name AS team,
                       s.season_name AS season,
                       count(DISTINCT f) AS games_played,
                       sum(r.goals_scored) AS total_goals,
                       sum(r.assists) AS total_assists,
                       sum(r.clean_sheets) AS total_clean_sheets,
                       sum(r.total_points) AS total_points,
                       avg(r.total_points) AS avg_points_per_game
            """)

        # -------------------------------------
        # Intent: Recommendation
        # -------------------------------------
        elif intent == "recommendation":
            # 1. Value Picks (Points per 90min)
            queries.append("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (p)-[:PLAYS_AS]->(pos:Position)
                WHERE ($position IS NULL OR toLower(pos.name) CONTAINS toLower($position))
                WITH p, pos, sum(r.total_points) as pts, sum(r.minutes) as mins
                WHERE mins > 500
                RETURN p.player_name AS player, 
                       pos.name AS position, 
                       pts AS total_points, 
                       (toFloat(pts)/mins * 90) as pts_per_90
                ORDER BY pts_per_90 DESC
                LIMIT 5
            """)

            # 2. Form (Last 3 Games)
            queries.append("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)
                WITH p, r, gw ORDER BY gw.GW_number DESC
                WITH p, collect(r.total_points)[0..3] as recent_points
                RETURN p.player_name AS player, 
                       reduce(s = 0, x IN recent_points | s + x) as form_score
                ORDER BY form_score DESC
                LIMIT 5
            """)

        # -------------------------------------
        # Run queries
        # -------------------------------------
        all_results = {}
        for i, cypher in enumerate(queries):
            params = {
                "player_name": entities.get("player_name", [None])[0] if entities.get("player_name") else None,
                "team": entities.get("team", [None])[0] if entities.get("team") else None,
                "position": entities.get("position", [None])[0] if entities.get("position") else None,
                "gameweek": entities.get("gameweek", [None])[0] if entities.get("gameweek") else None,
                "season": entities.get("season", [None])[0] if entities.get("season") else None
            }
            results = self._run_query(cypher, params)
            all_results[f"{intent}_{i+1}"] = results

        return all_results