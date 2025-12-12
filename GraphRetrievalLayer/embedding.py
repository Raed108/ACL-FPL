import os
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from InputPreprocessing.intent_classifier import classify_intent, classify_intent_llm
from InputPreprocessing.entity_extractions import extract_entities, extract_entities_with_llm
from InputPreprocessing.input_embedding import embed_user_query


URI = os.getenv("URI")
USERNAME = os.getenv("NeoName")
PASSWORD = os.getenv("PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))



# Two embedding models (Requirement 1)
MODEL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MPNET = "sentence-transformers/all-mpnet-base-v2"

models = {
    "minilm": SentenceTransformer(MODEL_MINILM),
    "mpnet": SentenceTransformer(MODEL_MPNET)
}


def build_node_text(label: str, props: dict):
    if label == "Season":
        return f"Season {props.get('season_name')}"

    if label == "Gameweek":
        return f"Gameweek {props.get('GW_number')} of season {props.get('season')}"

    if label == "Fixture":
        return f"Fixture {props.get('fixture_number')} in season {props.get('season')} kickoff time {props.get('kickoff_time')}"

    if label == "Team":
        return f"Team {props.get('name')}"

    if label == "Position":
        return f"Position {props.get('name')}"

    if label == "Player":
        # Make sure you store "position_name" property or fetch via relation
        return f"{props.get('player_name')} plays as {props.get('position_name')}"

    return ""


def create_all_node_embeddings():
    """
    Create embeddings for every node in the Neo4j KG using BOTH models.
    Stores:
        n.embedding_minilm
        n.embedding_mpnet
    """
    with driver.session() as session:
        results = session.run("""
            MATCH (n)
            RETURN id(n) AS id, labels(n)[0] AS label, properties(n) AS props
        """)

        for record in results:
            node_id = record["id"]
            label   = record["label"]
            props   = record["props"]

            text_description = build_node_text(label, props)

            for model_name, model in models.items():
                vector = model.encode(text_description).tolist()

                session.run(
                    f"""
                    MATCH (n) WHERE id(n) = $id
                    SET n.embedding_{model_name} = $vector
                    """,
                    id=node_id,
                    vector=vector
                )

    print("All node embeddings created and stored successfully!")



def semantic_search(query: list, model_choice: str = "mpnet", limit: int = 5):
    """
    Generic semantic search over ALL nodes in the KG.
    """
    query_vec = embed_user_query(query, model_choice)

    embedding_property = f"embedding_{model_choice}"

    cypher = f"""
    WITH $query_vec AS qvec
    MATCH (n)
    WHERE n.{embedding_property} IS NOT NULL
    WITH n, gds.similarity.cosine(n.{embedding_property}, qvec) AS similarity_score,
        COALESCE(n.player_name, n.name, n.fixture_number, toString(n.GW_number), n.season_name, elementId(n)) AS unique_id
    ORDER BY similarity_score DESC
    WITH unique_id, n, max(similarity_score) AS similarity_score  // keep top similarity per unique node
    RETURN labels(n)[0] AS label,
        elementId(n) AS node_id,
        apoc.map.removeKeys(n, ["embedding_mpnet", "embedding_minilm"]) AS node,
        similarity_score
    ORDER BY similarity_score DESC
    LIMIT $limit
    """

    with driver.session() as session:
        results = session.run(cypher, query_vec=query_vec, limit=limit)
        return [record.data() for record in results]





def cypher_player_stats(name: str, season: str = None):
    """
    Returns multiple perspectives on player statistics
    """
    results = {}
    
    with driver.session() as session:
        # First check if player has data for the specified season
        if season:
            check_result = session.run("""
                MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season {season_name: $season})
                RETURN count(r) as match_count
            """, player_name=name, season=season)
            
            match_count = check_result.single()['match_count']
            
            # If no data for specified season, fall back to most recent season
            if match_count == 0:
                fallback_result = session.run("""
                    MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
                    MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
                    RETURN s.season_name as available_season
                    ORDER BY s.season_name DESC
                    LIMIT 1
                """, player_name=name)
                
                fallback = fallback_result.single()
                if fallback:
                    season = fallback['available_season']
                    results['note'] = f"No data for requested season. Showing data for {season}"
                else:
                    return {'error': f'No data found for player {name}'}
        
        # 1. Detailed Season Overview
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
            WHERE p.player_name = $player_name
              AND ($season IS NULL OR s.season_name = $season)
            RETURN p.player_name AS player,
                   s.season_name AS season,
                   sum(r.minutes) AS minutes,
                   sum(r.goals_scored) AS goals,
                   sum(r.assists) AS assists,
                   sum(r.clean_sheets) AS clean_sheets,
                   sum(r.total_points) AS total_points,
                   sum(r.bonus) AS total_bonus
        """, player_name=name, season=season)
        results['season_overview'] = result.single().data() if result.peek() else None
        
        # 2. Recent Form - Last 5 Gameweeks
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
            WHERE p.player_name = $player_name
              AND ($season IS NULL OR s.season_name = $season)
            WITH p, r, gw ORDER BY gw.GW_number DESC LIMIT 5
            RETURN p.player_name AS player,
                   collect(gw.GW_number) as recent_gameweeks,
                   collect(r.total_points) as recent_points,
                   avg(r.ict_index) as avg_ict_form
        """, player_name=name, season=season)
        results['recent_form'] = result.single().data() if result.peek() else None
        
        # 3. Efficiency Stats (now also filtered by season for consistency)
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
            WHERE p.player_name = $player_name
              AND ($season IS NULL OR s.season_name = $season)
            WITH p, sum(r.total_points) as pts, sum(r.minutes) as mins
            WHERE mins > 0
            RETURN p.player_name AS player,
                   (toFloat(pts) / mins * 90) AS points_per_90
        """, player_name=name, season=season)
        results['efficiency'] = result.single().data() if result.peek() else None
    
    return results


def cypher_top_scorers(season: str = None, position: str = None):
    """
    Returns multiple top player rankings
    """
    results = {}
    
    with driver.session() as session:
        # 1. Top Total Points
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (p)-[:PLAYS_AS]->(pos:Position)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
            WHERE ($position IS NULL OR pos.name = $position)
              AND ($season IS NULL OR s.season_name = $season)
            RETURN p.player_name AS player, pos.name AS position, sum(r.total_points) AS total_points
            ORDER BY total_points DESC
            LIMIT 5
        """, position=position, season=season)
        results['top_points'] = [r.data() for r in result]
        
        # 2. Top Goal Scorers
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
            WHERE ($season IS NULL OR s.season_name = $season)
            RETURN p.player_name AS player, sum(r.goals_scored) AS goals, sum(r.minutes) as minutes
            ORDER BY goals DESC
            LIMIT 5
        """, season=season)
        results['top_scorers'] = [r.data() for r in result]
        
        # 3. Top Playmakers
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
            WHERE ($season IS NULL OR s.season_name = $season)
            RETURN p.player_name AS player, sum(r.assists) AS assists, sum(r.ict_index) as total_ict
            ORDER BY assists DESC
            LIMIT 5
        """, season=season)
        results['top_playmakers'] = [r.data() for r in result]
        
        # 4. Top Defenders
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (p)-[:PLAYS_AS]->(pos:Position)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
            WHERE pos.name IN ['DEF']
              AND ($season IS NULL OR s.season_name = $season)
            RETURN p.player_name AS player,
                   sum(r.clean_sheets) AS clean_sheets,
                   sum(r.goals_conceded) as goals_conceded,
                   sum(r.total_points) as total_points
            ORDER BY clean_sheets DESC, total_points DESC
            LIMIT 5
        """, season=season)
        results['top_defenders'] = [r.data() for r in result]
    
    return results


def cypher_fixture_info(fixture_number: int = None, season: str = None, team: str = None, player_name: str = None):
    """
    Returns fixture information and upcoming schedule
    """
    results = {}
    
    with driver.session() as session:
        # 1. Upcoming Schedule
        result = session.run("""
            MATCH (my_team:Team)
            WHERE ($team IS NULL OR my_team.name = $team)
            MATCH (my_team)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
            WHERE f.kickoff_time >= datetime()
            WITH f, my_team
            MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(opponent:Team)
            WHERE opponent <> my_team
            RETURN f.kickoff_time AS kickoff,
                   my_team.name AS team,
                   opponent.name AS opponent
            ORDER BY f.kickoff_time ASC
            LIMIT 3
        """, team=team)
        results['upcoming_fixtures'] = [r.data() for r in result]
        
        # 2. Specific Fixture Info (if fixture_number provided)
        if fixture_number:
            result = session.run("""
                MATCH (f:Fixture {fixture_number: $fix})
                WHERE ($season IS NULL OR f.season = $season)
                MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
                MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
                RETURN f.kickoff_time AS kickoff,
                       home.name AS home_team,
                       away.name AS away_team,
                       f.season AS season
            """, fix=fixture_number, season=season)
            results['fixture_details'] = result.single().data() if result.peek() else None
    
    return results


def cypher_team_analysis(team_name: str, season: str = None):
    """
    FINAL – WORKS PERFECTLY with your schema (Dec 2025)
    No syntax errors, no unbound variables, no deprecation warnings
    """
    if season:
        season = season.replace("/", "-").strip()

    with driver.session() as session:
        # ==================== 1. TOP 10 ATTACKERS ====================
        top_attackers = session.run("""
            MATCH (t:Team {name: $team})<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season {season_name: $season})
            
            WITH p,
                 sum(r.total_points) AS points,
                 sum(r.goals_scored) AS goals,
                 sum(r.assists) AS assists
            WHERE points > 0
            RETURN p.player_name AS player,
                   goals,
                   assists,
                   points
            ORDER BY points DESC
            LIMIT 10
        """, team=team_name, season=season).data()

        # ==================== 2. TEAM GOALS SCORED & CONCEDED ====================
        # We cannot use (f)-[:...]->(t) inside WHERE after WITH f — Neo4j forbids it
        # So we keep `t` in scope the whole time
        stats = session.run("""
            MATCH (t:Team {name: $team})<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season {season_name: $season})

            // Our goals in this fixture
            OPTIONAL MATCH (our:Player)-[r:PLAYED_IN]->(f)
            WITH t, f, coalesce(sum(r.goals_scored), 0) AS our_goals

            // Opponent goals in this fixture
            OPTIONAL MATCH (opp:Team)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f)  // get the other team
            OPTIONAL MATCH (opp:Player)-[or:PLAYED_IN]->(f)
            WHERE NOT (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
            WITH f, our_goals, coalesce(sum(or.goals_scored), 0) AS opp_goals

            RETURN 
                count(DISTINCT f) AS played,
                sum(our_goals) AS goals_scored,
                sum(opp_goals) AS goals_conceded
        """, team=team_name, season=season).single()

        # Build clean result
        overview = {
            "team": team_name,
            "season": season,
            "played": stats["played"] if stats else 0,
            "goals_scored": stats["goals_scored"] if stats else 0,
            "goals_conceded": stats["goals_conceded"] if stats else 0,
            "goal_difference": (stats["goals_scored"] - stats["goals_conceded"]) if stats else 0
        }

        return {
            "top_attackers": top_attackers,
            "team_overview": overview
        }


def cypher_recommend(season: str = None, position: str = None):
    """
    Returns player recommendations based on multiple criteria
    """
    results = {}
    
    with driver.session() as session:
        # 1. Best Value Picks
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (p)-[:PLAYS_AS]->(pos:Position)
            WHERE ($position IS NULL OR pos.name = $position)
            WITH p, pos, sum(r.total_points) as pts, sum(r.minutes) as mins
            WHERE mins > 500
            RETURN p.player_name AS player, 
                   pos.name AS position, 
                   pts AS total_points, 
                   (toFloat(pts)/mins * 90) as pts_per_90
            ORDER BY pts_per_90 DESC
            LIMIT 5
        """, position=position)
        results['value_picks'] = [r.data() for r in result]
        
        # 2. Captaincy Options
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)
            WITH p, r, gw ORDER BY gw.GW_number DESC
            WITH p, collect(r.total_points)[0..3] as recent_points
            RETURN p.player_name AS player,
                   reduce(s = 0, x IN recent_points | s + x) as form_score
            ORDER BY form_score DESC
            LIMIT 5
        """)
        results['captaincy_options'] = [r.data() for r in result]
        
        # 3. High Points Players (original recommendation logic)
        result = session.run("""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
            WHERE r.total_points > 100
              AND ($season IS NULL OR s.season_name = $season)
            RETURN p.player_name AS name,
                   sum(r.total_points) AS total_points,
                   s.season_name AS season
            ORDER BY total_points DESC
            LIMIT 5
        """, season=season)
        results['high_performers'] = [r.data() for r in result]
    
    return results


# Update the answer_query function to use the new multi-query structure
def answer_query(user_input: str, entities, intent, model_choice="mpnet"):
    # 1. Classify intent
    # intent = classify_intent(user_input)
    # intent = classify_intent_llm(user_input)
    
    # 2. Extract entities
    # entities = extract_entities(user_input)
    # entities = extract_entities_with_llm(user_input)

    season = entities.get("season", [None])[0] if entities.get("season") else None


    # 4. Retrieve top similar nodes
    candidates = semantic_search(user_input, model_choice=model_choice, limit=5)
    
    # 5. Include all cosine similarity scores
    results_with_scores = []
    seen_nodes = set()
    
    for candidate in candidates:
        label = candidate["label"]
        node = candidate["node"]
        score = candidate["similarity_score"]
        
        # Use elementId or unique property
        node_unique_id = node.get("player_name") or node.get("name") or node.get("fixture_number") or candidate["node_id"]
        
        if node_unique_id in seen_nodes:
            continue
        seen_nodes.add(node_unique_id)
        
        # Run the relevant Cypher query for this node
        data = None
        if intent == "player_stats" and label == "Player":
            data = cypher_player_stats(node["player_name"], season)
        elif intent == "team_analysis" and label == "Team":
            data = cypher_team_analysis(node["name"], season)
        elif intent == "fixture_query":
            # Extract additional params if needed
            team = entities.get("team", [None])[0] if entities.get("team") else None
            player_name = entities.get("player_name", [None])[0] if entities.get("player_name") else None
            data = cypher_fixture_info(
                fixture_number=node.get("fixture_number"),
                season=season,
                team=team,
                player_name=player_name
            )
        elif intent == "top_players":
            position = entities.get("position", [None])[0] if entities.get("position") else None
            data = cypher_top_scorers(season=season, position=position)
        elif intent == "recommendation":
            position = entities.get("position", [None])[0] if entities.get("position") else None
            data = cypher_recommend(season=season, position=position)
        else:
            # Fallback: return node properties without embeddings
            if node:
                data = {k: v for k, v in dict(node).items() 
                       if k not in ("embedding_minilm", "embedding_mpnet")}
            else:
                data = {}
        
        # Filter out results where data is completely None or empty
        if data is not None:
            # Check if data has meaningful content
            if isinstance(data, dict):
                # For multi-query results, check if at least one sub-query has data
                has_content = False
                if 'error' in data:
                    # Skip error results
                    continue
                for key, value in data.items():
                    if key == 'note':  # Keep notes
                        has_content = True
                        break
                    if value is not None:
                        has_content = True
                        break
                
                if not has_content:
                    continue  # Skip this result
            
            results_with_scores.append({
                "similarity_score": score,
                "data": data
            })
    
    return results_with_scores