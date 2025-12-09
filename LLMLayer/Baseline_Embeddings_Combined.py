def combine_retrieval_results(baseline_results, vector_results):
    """
    Combines results from Cypher queries and Vector search.
    Assumes results are lists of dictionaries containing node data.
    """
    combined_context = []
    seen_ids = set()

    # Process Baseline Results (Cypher)
    for item in baseline_results:
        # Assuming each item has a unique identifier, e.g., 'id' or 'name'
        unique_id = item.get('id') or item.get('name')
        if unique_id not in seen_ids:
            combined_context.append(item)
            seen_ids.add(unique_id)

    # Process Vector Results (Embeddings)
    for item in vector_results:
        unique_id = item.get('id') or item.get('name')
        if unique_id not in seen_ids:
            combined_context.append(item)
            seen_ids.add(unique_id)
            
    # Optional: Rank results here if necessary
    
    return combined_context