def combine_retrieval_results(baseline_results=None, vector_results=None, hybrid_results=None):
    """
    Combines results from Cypher queries, Vector search, and Hybrid retrieval.
    Deduplicates results using player names or node IDs.
    """
    combined_context = []
    seen_ids = set()

    # --- Helper: Adds a single item to context with deduplication ---
    def add_single_item(item, category_name=None):
        if not isinstance(item, dict):
            return

        # STRATEGY: Use 'player', 'name', or 'id' as Unique ID
        unique_id = item.get('player') or item.get('name') or item.get('id')
        
        # Fallback
        if not unique_id:
            unique_id = str(item)

        # Unique ID Construction for specific stats (optional but helpful)
        # If we have distinct stats like 'season_overview' vs 'recent_form', 
        # we might want to keep both. For now, we dedup by player name to keep context clean.
        
        # Logic: If we haven't seen this ID, add it.
        # Note: In your second example, "season_overview" and "recent_form" describe the SAME player.
        # Usually, you want to merge these or include both. 
        # To keep it simple: we treat the inner dictionary (e.g. season_overview) as the item.
        
        # IMPROVEMENT: If the item is just a part of a larger player profile, allow it if it provides new data.
        # But to match previous logic, we check strictly by ID.
        if unique_id not in seen_ids:
            if category_name:
                item['retrieval_source'] = category_name
            combined_context.append(item)
            seen_ids.add(unique_id)
        else:
            # If ID is seen, but this is a different category (e.g. form vs overview),
            # we might want to append it to the existing record in a real app.
            # For this specific prompt requirement, we just add it to context list 
            # effectively allowing "Mohamed Salah (Overview)" and "Mohamed Salah (Form)"
            # if we make the ID unique per category.
            
            # For now, let's append it to context directly if it's a valid data piece
            # that helps the LLM, even if the player name is the same.
            # To do this safely, we make a composite ID:
            composite_id = f"{unique_id}_{category_name}" if category_name else unique_id
            if composite_id not in seen_ids:
                if category_name:
                    item['retrieval_source'] = category_name
                combined_context.append(item)
                seen_ids.add(composite_id)

    # --- 1. Process Baseline Results (Standard Dictionary) ---
    if baseline_results and isinstance(baseline_results, dict):
        for category, items in baseline_results.items():
            if isinstance(items, list):
                for item in items:
                    add_single_item(item, category)

    # --- 2. Process Hybrid Results (List of Wrappers) ---
    if hybrid_results and isinstance(hybrid_results, list):
        for wrapper in hybrid_results:
            data_payload = wrapper.get('data', {})
            
            if isinstance(data_payload, dict):
                for key, value in data_payload.items():
                    
                    # CASE A: Value is a LIST (e.g., 'top_points': [player1, player2])
                    if isinstance(value, list):
                        for item in value:
                            add_single_item(item, key)
                            
                    # CASE B: Value is a DICT (e.g., 'season_overview': {player: 'Salah', ...})
                    elif isinstance(value, dict):
                        add_single_item(value, key)

    # --- 3. Process Vector Results (List of Node Dictionaries) ---
    if vector_results and isinstance(vector_results, list):
        for item in vector_results:
            node_data = item.get('node', {}).copy()
            unique_id = node_data.get('player') or node_data.get('name') or item.get('node_id')
            
            if unique_id and unique_id not in seen_ids:
                if 'label' in item:
                    node_data['entity_type'] = item['label']
                combined_context.append(node_data)
                seen_ids.add(unique_id)

    return combined_context