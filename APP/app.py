import streamlit as st
import pandas as pd
import sys
import os
import time
import requests
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

# --- Load ENV ---
load_dotenv()

# --- User Modules ---
from InputPreprocessing.intent_classifier import classify_intent
from InputPreprocessing.entity_extractions import extract_entities
from GraphRetrievalLayer.Baseline import GraphRetrieval
from LLMLayer.Baseline_Embeddings_Combined import combine_retrieval_results

# --- OpenRouter Model Config ---
MODELS = {
    "llama3": "meta-llama/llama-3-8b-instruct",
    "gemma2": "google/gemma-2-9b-it",
    "mistralai": "mistralai/mistral-7b-instruct:free"
}
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Streamlit Config ---
st.set_page_config(
    page_title="FPL Analytics | AI-Powered Insights",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Theme CSS ---
st.markdown("""
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    :root {
        --white: #F6F7ED;
        --neon-yellow: #DBE64C;
        --dark-blue: #001F3F;
        --mantis: #74C365;
        --forest-green: #00804C;
        --secondary-blue: #1E488F;
    }
    
    .stApp {
        background: linear-gradient(135deg, #001F3F 0%, #0a2d5a 50%, #001F3F 100%);
        color: #F6F7ED;
    }
    
    /* Animated background texture */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(219, 230, 76, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(116, 195, 101, 0.03) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Main container */
    .main {
        background: transparent;
        padding: 2rem 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1f35 0%, #001F3F 100%);
        border-right: 2px solid rgba(219, 230, 76, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #F6F7ED !important;
    }
    
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #DBE64C !important;
        font-weight: 500;
    }
    
    /* Headings */
    h1 {
        color: #DBE64C;
        font-size: 2.5rem;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(219, 230, 76, 0.3);
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    
    h2, h3 {
        color: #DBE64C;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(219, 230, 76, 0.2);
    }
    
    /* Neon card styling */
    .neon-card {
        background: linear-gradient(135deg, rgba(10, 45, 90, 0.6) 0%, rgba(0, 31, 63, 0.8) 100%);
        border: 1.5px solid rgba(219, 230, 76, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(219, 230, 76, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .neon-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(219, 230, 76, 0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .neon-card:hover {
        border-color: rgba(219, 230, 76, 0.6);
        box-shadow: 0 12px 48px rgba(219, 230, 76, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.2);
        transform: translateY(-4px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #74C365 0%, #5a9f4f 100%);
        color: #001F3F;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(116, 195, 101, 0.2);
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #DBE64C 0%, #b8d43d 100%);
        box-shadow: 0 8px 25px rgba(219, 230, 76, 0.4);
        transform: translateY(-2px);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        background: rgba(15, 35, 60, 0.8) !important;
        border: 1.5px solid rgba(116, 195, 101, 0.3) !important;
        border-radius: 12px !important;
        color: #F6F7ED !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #DBE64C !important;
        box-shadow: 0 0 15px rgba(219, 230, 76, 0.3) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(219, 230, 76, 0.2);
        background: linear-gradient(135deg, rgba(10, 45, 90, 0.5) 0%, rgba(0, 31, 63, 0.7) 100%);
    }
    
    /* Tabs */
    .stTabs [role="tab"] {
        border-radius: 12px 12px 0 0;
        border: 1px solid rgba(116, 195, 101, 0.2);
        color: #DBE64C;
        font-weight: 600;
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        border-color: #DBE64C;
        background: rgba(219, 230, 76, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(116, 195, 101, 0.1) 0%, rgba(219, 230, 76, 0.05) 100%);
        border-radius: 12px;
        border: 1px solid rgba(116, 195, 101, 0.2);
        color: #DBE64C !important;
        font-weight: 600;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 128, 76, 0.2) 0%, rgba(116, 195, 101, 0.1) 100%);
        border: 1px solid rgba(116, 195, 101, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        color: #DBE64C;
        font-size: 1.8rem;
        font-weight: 900;
    }
    
    .metric-label {
        color: rgba(246, 247, 237, 0.7);
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Info/Warning boxes */
    .stInfo, .stWarning, .stSuccess {
        border-radius: 12px;
        border: 1.5px solid rgba(219, 230, 76, 0.4);
        background: rgba(10, 45, 90, 0.6) !important;
        color: #F6F7ED !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 31, 63, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(116, 195, 101, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(219, 230, 76, 0.6);
    }
    
    /* Spinner animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .spinner-text {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions (UNCHANGED LOGIC) ---
def generate_llm_response(model_name, context, user_query, intent):
    """Generate LLM response using OpenRouter API."""
    system_prompt = f"""
You are an expert Fantasy Premier League (FPL) Assistant.

Your Task: Answer the user's question strictly based on the provided Context.
User Intent: {intent}

Context:
{context}

Guidelines:
1. Be helpful, concise, and enthusiastic (like a football commentator).
2. If the answer is not in the context, admit you don't know.
3. Use bolding for player names and key stats.
4. If recommending players, explain WHY based on stats.
"""
    model_id = MODELS.get(model_name)
    if not model_id:
        return "⚠️ Selected model is not available."

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": f"User Query: {user_query}\n\nSystem Instructions: {system_prompt}"}]
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

    try:
        start_time = time.time()
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
        data = response.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "No answer returned.")
        return answer
    except Exception as e:
        return f"❌ Error calling model: {str(e)}"

def format_context_for_display(results):
    """Format context results for display."""
    if isinstance(results, list):
        return results
    formatted_list = []
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, list):
                formatted_list.extend(value)
            else:
                formatted_list.append(value)
    return formatted_list

def visualize_graph(baseline_results):
    """Create a NetworkX graph visualization from KG results."""
    G = nx.Graph()
    for key, data_list in baseline_results.items():
        for item in data_list:
            if isinstance(item, dict):
                nodes = list(item.keys())
                for i in range(len(nodes)-1):
                    G.add_edge(nodes[i], nodes[i+1])
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#001F3F')
    
    nx.draw_networkx_nodes(G, pos, node_color='#74C365', node_size=1200, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='#DBE64C', width=2, ax=ax, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_color='#001F3F', font_weight='bold', font_size=9, ax=ax)
    
    ax.set_facecolor('#001F3F')
    ax.axis('off')
    st.pyplot(fig)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚽</div>
        <h2 style="margin: 0; font-size: 1.3rem; text-shadow: 0 0 15px rgba(219, 230, 76, 0.4);">FPL Analytics</h2>
        <p style="color: rgba(246, 247, 237, 0.7); font-size: 0.85rem; margin-top: 0.25rem;">AI-Powered Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<h3 style='color: #DBE64C;'>⚙️ Configuration</h3>", unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "🧠 Select LLM Model",
        ["llama3", "gemma2", "mistralai"],
        index=0,
        help="Choose the AI model for generating insights"
    )
    
    retrieval_method = st.radio(
        "🔍 Retrieval Strategy",
        ["Baseline Only", "Embedding Only", "Baseline + Embedding"],
        index=2,
        help="Select how to retrieve and combine results"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div class="neon-card" style="margin-top: 1rem;">
        <div style="color: #DBE64C; font-weight: 700; margin-bottom: 0.5rem;">💡 Pro Tip</div>
        <div style="font-size: 0.9rem; color: rgba(246, 247, 237, 0.85); line-height: 1.5;">
            <strong>Baseline + Embedding</strong> combines exact keyword matches with semantic search for superior accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("📊 FPL Graph-RAG v1.0 | Powered by AI")

# --- Main Interface ---
st.markdown("""
<div style="margin-bottom: 0.5rem;">
    <h1 style="display: inline-block; margin-right: 1rem;">⚽ FPL Analytics</h1>
    <span style="color: #74C365; font-size: 1rem; vertical-align: middle;">AI-Powered Insights</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="color: rgba(246, 247, 237, 0.8); font-size: 1.05rem; margin-bottom: 1.5rem;">
    Ask about <strong style="color: #74C365;">Player Stats</strong> • <strong style="color: #74C365;">Fixtures</strong> • 
    <strong style="color: #74C365;">Team Analysis</strong> • <strong style="color: #74C365;">Fantasy Recommendations</strong>
</p>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello Manager! Welcome to FPL Analytics. I'm here to help you dominate your Fantasy Premier League season with AI-powered insights. Who are we analyzing today?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Processing ---
if prompt := st.chat_input("🔍 Ask about players, fixtures, tactics, or get recommendations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🔎 Scouting the database..."):
            try:
                # Preprocessing (UNCHANGED)
                intent = classify_intent(prompt)
                entities = extract_entities(prompt)

                # Retrieval (UNCHANGED)
                baseline_results = {}
                vector_results = []

                if retrieval_method in ["Baseline Only", "Baseline + Embedding"]:
                    graph_retriever = GraphRetrieval()
                    baseline_results = graph_retriever.retrieve_kg_context(entities, intent)
                    graph_retriever.close()

                if retrieval_method in ["Embedding Only", "Baseline + Embedding"]:
                    vector_results = []

                # Combine Results (UNCHANGED)
                if retrieval_method == "Baseline + Embedding":
                    flattened_baseline = format_context_for_display(baseline_results)
                    combined_context = combine_retrieval_results(flattened_baseline, vector_results)
                elif retrieval_method == "Baseline Only":
                    combined_context = format_context_for_display(baseline_results)
                else:
                    combined_context = vector_results

                context_str = "\n".join([str(item) for item in combined_context])
                if not context_str:
                    context_str = "No specific data found in the Knowledge Graph for this query."

                # LLM Response (UNCHANGED)
                full_response = generate_llm_response(model_choice, context_str, prompt, intent)
                message_placeholder.markdown(full_response)

                # --- Transparency UI ---
                with st.expander("🧐 View Retrieval Details"):
                    tab1, tab2, tab3, tab4 = st.tabs(["Entities & Intent", "Context", "Queries", "Graph"])
                    
                    with tab1:
                        st.markdown("<h4 style='color: #DBE64C;'>Detected Intent</h4>", unsafe_allow_html=True)
                        st.code(intent, language="text")
                        st.markdown("<h4 style='color: #DBE64C; margin-top: 1rem;'>Extracted Entities</h4>", unsafe_allow_html=True)
                        st.json(entities)
                    
                    with tab2:
                        st.markdown("<h4 style='color: #DBE64C;'>Context for LLM</h4>", unsafe_allow_html=True)
                        st.text_area("", context_str, height=250, disabled=True)

                    with tab3:
                        st.markdown("<h4 style='color: #DBE64C;'>Cypher Queries</h4>", unsafe_allow_html=True)
                        if baseline_results:
                            for query_key in baseline_results.keys():
                                st.code(f"MATCH (n)-[r]->(m) WHERE ... RETURN ...  // {query_key}", language="cypher")
                        else:
                            st.info("No Cypher queries executed for this request.")
                    
                    with tab4:
                        st.markdown("<h4 style='color: #DBE64C;'>Knowledge Graph Visualization</h4>", unsafe_allow_html=True)
                        if baseline_results:
                            visualize_graph(baseline_results)
                        else:
                            st.info("No graph data available for visualization.")

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
                full_response = "I encountered an error processing your request. Please try again."

    st.session_state.messages.append({"role": "assistant", "content": full_response})
