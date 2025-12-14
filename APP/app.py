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
from InputPreprocessing.intent_classifier import classify_intent, classify_intent_llm
from InputPreprocessing.entity_extractions import extract_entities, extract_entities_with_llm
from GraphRetrievalLayer.Baseline import GraphRetrieval
from GraphRetrievalLayer.embedding import answer_query, semantic_search
from LLMLayer.Baseline_Embeddings_Combined import combine_retrieval_results
from LLMLayer.Prompt_Structure import create_prompt_template
from Model_Evaluation.model_evaluator import query_llm

# --- OpenRouter Model Config ---
MODELS = {
    "llama3.3": "meta-llama/llama-3.3-70b-instruct:free",
    "gemma3": "google/gemma-3-12b-it:free",
    "mistralai": "mistralai/mistral-7b-instruct:free"
}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Streamlit Config ---
st.set_page_config(
    page_title="FPL COMMAND CENTER | AI Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* Chat input wrapper */
div[data-testid="stChatInput"] {
    position: relative;
}

/* Floating icon inside input */
.chat-question-icon {
    position: absolute;
    right: 14px;
    bottom: 14px;
    z-index: 1001;
}

/* Icon button style */
.chat-question-icon button {
    background: transparent !important;
    border: none !important;
    font-size: 1.35rem;
    cursor: pointer;
    padding: 0;
    color: #DBE64C;
}

/* Prevent text overlap with icon */
div[data-testid="stChatInput"] textarea {
    padding-right: 3rem !important;
}
</style>
""", unsafe_allow_html=True)

# --- Global Theme CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * { 
        margin: 0; 
        padding: 0; 
        box-sizing: border-box;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    :root {
        --white: #F6F7ED;
        --neon-yellow: #DBE64C;
        --dark-blue: #001F3F;
        --mantis: #74C365;
        --forest-green: #00804C;
        --navy: #0a1f35;
    }
    
    /* Main Background - Deep Tech Feel */
    .stApp {
        background: #000000;
        color: #F6F7ED;
    }
    
    /* Animated Grid Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(219, 230, 76, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(219, 230, 76, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
        animation: gridPulse 20s ease-in-out infinite;
    }
    
    @keyframes gridPulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
    
    /* Dynamic Gradient Overlay */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse at top left, rgba(219, 230, 76, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(116, 195, 101, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at center, rgba(0, 31, 63, 0.4) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Main Container */
    .main {
        background: transparent;
        padding: 1.5rem 1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar - Command Panel Style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000 0%, #001F3F 100%);
        border-right: 3px solid #DBE64C;
        box-shadow: 4px 0 20px rgba(219, 230, 76, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #F6F7ED !important;
    }
    
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #DBE64C !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.75rem;
    }
    
    /* Header Styles - Bold & Athletic */
    h1 {
        color: #F6F7ED;
        font-size: 3.5rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: -1px;
        margin-bottom: 0rem;
        line-height: 1;
        text-shadow: 0 0 40px rgba(219, 230, 76, 0.4);
        position: relative;
    }
    
    h1::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 0;
        width: 120px;
        height: 4px;
        background: linear-gradient(90deg, #DBE64C 0%, transparent 100%);
    }
    
    h2 {
        color: #DBE64C;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 1.1rem;
    }
    
    h3 {
        color: #74C365;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    h4 {
        color: #DBE64C;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Premium Card Design */
    .neon-card {
        background: linear-gradient(135deg, rgba(0, 31, 63, 0.4) 0%, rgba(0, 0, 0, 0.6) 100%);
        border: 2px solid rgba(219, 230, 76, 0.4);
        border-radius: 20px;
        padding: 1.75rem;
        backdrop-filter: blur(20px);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(219, 230, 76, 0.2),
            0 0 40px rgba(219, 230, 76, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .neon-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(219, 230, 76, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .neon-card:hover::before {
        left: 100%;
    }
    
    .neon-card:hover {
        border-color: #DBE64C;
        box-shadow: 
            0 12px 48px rgba(219, 230, 76, 0.3),
            inset 0 1px 0 rgba(219, 230, 76, 0.3),
            0 0 60px rgba(219, 230, 76, 0.2);
        transform: translateY(-6px);
    }
    
    /* Stat Card */
    .stat-card {
        background: linear-gradient(135deg, rgba(116, 195, 101, 0.15) 0%, rgba(0, 128, 76, 0.1) 100%);
        border: 2px solid rgba(116, 195, 101, 0.5);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .stat-card::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(219, 230, 76, 0.15) 0%, transparent 70%);
    }
    
    /* Buttons - Power Style */
    .stButton > button {
        background: linear-gradient(135deg, #74C365 0%, #00804C 100%);
        color: #000000;
        border: none;
        border-radius: 14px;
        padding: 1rem 2rem;
        font-weight: 800;
        font-size: 0.9rem;
        letter-spacing: 1px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 4px 20px rgba(116, 195, 101, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #DBE64C 0%, #74C365 100%);
        box-shadow: 
            0 8px 32px rgba(219, 230, 76, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        transform: translateY(-3px) scale(1.02);
    }
    
    /* Input Fields - High-Tech */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        background: rgba(0, 0, 0, 0.6) !important;
        border: 2px solid rgba(116, 195, 101, 0.4) !important;
        border-radius: 14px !important;
        color: #F6F7ED !important;
        padding: 1rem 1.25rem !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #DBE64C !important;
        box-shadow: 0 0 20px rgba(219, 230, 76, 0.4) !important;
        background: rgba(0, 31, 63, 0.4) !important;
    }
    
    /* Chat Messages - Enhanced */
    .stChatMessage {
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
        border: 2px solid rgba(219, 230, 76, 0.3);
        background: linear-gradient(135deg, rgba(0, 31, 63, 0.3) 0%, rgba(0, 0, 0, 0.5) 100%);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .stChatMessage[data-testid="user-message"] {
        border-color: rgba(116, 195, 101, 0.5);
        background: linear-gradient(135deg, rgba(116, 195, 101, 0.15) 0%, rgba(0, 128, 76, 0.1) 100%);
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        border-color: rgba(219, 230, 76, 0.5);
        background: linear-gradient(135deg, rgba(219, 230, 76, 0.1) 0%, rgba(0, 31, 63, 0.2) 100%);
    }
    
    /* Tabs - Modern */
    .stTabs [role="tab"] {
        border-radius: 12px 12px 0 0;
        border: 2px solid rgba(116, 195, 101, 0.3);
        color: #F6F7ED;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [role="tab"]:hover {
        background: rgba(219, 230, 76, 0.1);
        border-color: rgba(219, 230, 76, 0.5);
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        border-color: #DBE64C;
        background: linear-gradient(180deg, rgba(219, 230, 76, 0.2) 0%, rgba(219, 230, 76, 0.05) 100%);
        color: #DBE64C;
        box-shadow: 0 -2px 10px rgba(219, 230, 76, 0.3);
    }
    
    /* Expander - Command Style */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(116, 195, 101, 0.15) 0%, rgba(0, 0, 0, 0.3) 100%);
        border-radius: 14px;
        border: 2px solid rgba(116, 195, 101, 0.4);
        color: #DBE64C !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 1rem 1.25rem;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #DBE64C;
        background: linear-gradient(135deg, rgba(219, 230, 76, 0.15) 0%, rgba(116, 195, 101, 0.1) 100%);
        box-shadow: 0 4px 15px rgba(219, 230, 76, 0.2);
    }
    
    /* Metrics Display */
    .metric-value {
        color: #DBE64C;
        font-size: 2.5rem;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(219, 230, 76, 0.5);
        line-height: 1;
    }
    
    .metric-label {
        color: rgba(246, 247, 237, 0.6);
        font-size: 0.75rem;
        margin-top: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Alert Boxes */
    .stInfo, .stWarning, .stSuccess, .stError {
        border-radius: 14px;
        border: 2px solid rgba(219, 230, 76, 0.5);
        background: rgba(0, 31, 63, 0.4) !important;
        color: #F6F7ED !important;
        backdrop-filter: blur(10px);
        padding: 1rem 1.25rem;
    }
    
    /* Scrollbar - Sleek */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #74C365 0%, #00804C 100%);
        border-radius: 10px;
        border: 2px solid rgba(0, 0, 0, 0.3);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #DBE64C 0%, #74C365 100%);
    }
    
    /* Loading Spinner */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
    }
    
    .spinner-text {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        color: #DBE64C;
        font-weight: 700;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #74C365 0%, #00804C 100%);
        color: #000000;
        box-shadow: 0 2px 10px rgba(116, 195, 101, 0.4);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(219, 230, 76, 0.5), transparent);
        margin: 1.5rem 0;
    }
    
    /* Code Blocks */
    code {
        background: rgba(0, 0, 0, 0.6) !important;
        border: 1px solid rgba(116, 195, 101, 0.3) !important;
        border-radius: 6px !important;
        color: #74C365 !important;
        padding: 0.25rem 0.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: #DBE64C !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.5px !important;
    }
    
    .stRadio > div {
        background: rgba(0, 31, 63, 0.3);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(116, 195, 101, 0.2);
    }
    
    /* Select Box */
    .stSelectbox > label {
        color: #DBE64C !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.5px !important;
    }
            
   /* REMOVE ALL SIDEBAR TOP SPACING (strongest override) */
    [data-testid="stSidebar"] {
        padding-top: 0 !important;
    }

    /* REMOVE INTERNAL CONTAINER SPACING */
    [data-testid="stSidebar"] .block-container {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* REMOVE EXTRA EMPTY DIVS STREAMLIT INJECTS */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* HEADER ITSELF — ultra-tight */
    .sidebar-header-tight {
        margin: 0 !important;
        padding: 0.2rem 0 !important; /* reduce this further if needed */
        line-height: 1 !important;
    }

    /* OPTIONAL: shrink icon */
    .sidebar-header-tight img,
    .sidebar-header-tight svg {
        width: 22px !important;
        height: 22px !important;
        vertical-align: middle !important;
    }
            
    /* Reduce space above the Pro Insight card */
    .pro-insight {
        margin-top: -40px !important;   /* Try -10, -20, -30 depending on how high you want it */
    }

     /* Add margin below selectbox and radio buttons */
    .stSelectbox, 
    .stRadio {
        margin-bottom: 0.25rem !important; /* adjust the value as needed */
    }


</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Fixed Quick Question Icon */
.fixed-question-icon {
    position: fixed;
    bottom: 90px;               /* aligns with chat_input */
    right: 30px;
    z-index: 9999;
}

/* Make icon visually part of input */
.fixed-question-icon button {
    border-radius: 50%;
    width: 38px;
    height: 38px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)


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

def normalize_baseline_results(results):
    """
    Normalize baseline results so all player dicts have the same keys.
    Missing keys are filled with None.
    """
    normalized_list = []

    if not isinstance(results, dict):
        return results  # fallback if results are not a dict

    for key, data_list in results.items():
        for r in data_list:
            if isinstance(r, dict):
                normalized = {
                    "player": r.get("player"),
                    "position": r.get("position"),
                    "total_points": r.get("total_points"),
                    "pts_per_90": r.get("pts_per_90"),
                    "form_score": r.get("form_score")
                }
                normalized_list.append(normalized)
    return normalized_list

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
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#000000')
    
    nx.draw_networkx_nodes(G, pos, node_color='#74C365', node_size=1500, ax=ax, alpha=0.9, 
                          edgecolors='#DBE64C', linewidths=2)
    nx.draw_networkx_edges(G, pos, edge_color='#DBE64C', width=3, ax=ax, alpha=0.7, 
                          style='solid')
    nx.draw_networkx_labels(G, pos, font_color='#F6F8ED', font_weight='bold', 
                           font_size=10, ax=ax)
    
    ax.set_facecolor('#000000')
    ax.axis('off')
    fig.tight_layout()
    st.pyplot(fig)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header-tight">
    <span style="font-size: 1.1rem; font-weight: 800; color: #DBE64C;">
        ⚽ FPL COMMAND CENTER
    </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='margin-bottom: 1.25rem; font-size: 1rem;'>⚙️ SYSTEM CONFIG</h2>", unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "🧠 AI MODEL",
        ["llama3.3", "gemma3", "mistralai"],
        index=0,
        help="Select the AI engine for analysis"
    )
    
    retrieval_method = st.radio(
        "🔍 RETRIEVAL MODE",
        ["Baseline Only", "Embedding Only", "Baseline + Embedding"],
        index=2,
        help="Choose data retrieval strategy"
    )
    
    # --- Show embedding model ONLY if needed ---
    if retrieval_method in ["Embedding Only", "Baseline + Embedding"]:
        embed_model = st.selectbox(
            "🧩 EMBEDDING ENGINE",
            ["minilm", "mpnet"],
            index=0,
            help="Semantic search model"
        )
    else:
        embed_model = None

    st.markdown("<hr style='margin-top: 1rem;>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="neon-card" style="margin-top: -0.75rem;">
        <div style="color: #DBE64C; font-weight: 800; margin-bottom: 0.75rem; 
                    font-size: 0.8rem; text-transform: uppercase; 
                    letter-spacing: 0.5px; margin-top: -0.7rem;">
            💡 PRO INSIGHT
        </div>
        <div style="font-size: 0.85rem; color: rgba(246, 247, 237, 0.8); 
                    line-height: 1.6; font-weight: 500;">
            <strong style="color: #74C365;">Baseline + Embedding</strong> mode delivers maximum accuracy by combining keyword precision with semantic intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown("<hr style='margin-top: 2rem;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 1.5rem;">
        <div class="status-badge">SYSTEM ACTIVE</div>
        <p style="color: rgba(246, 247, 237, 0.5); font-size: 0.7rem; margin-top: 1rem; font-weight: 600;">GRAPH-RAG v1.0</p>
    </div>
    """, unsafe_allow_html=True)

# --- Main Interface ---
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <h1 style="margin: 0;">FPL ANALYTICS</h1>
        <span class="status-badge">LIVE</span>
    </div>
    <p style="color: rgba(246, 247, 237, 0.7); font-size: 1.1rem; font-weight: 500; line-height: 1.5;">
        Advanced AI-powered intelligence for <span style="color: #74C365; font-weight: 700;">Player Performance</span> • 
        <span style="color: #74C365; font-weight: 700;">Match Fixtures</span> • 
        <span style="color: #74C365; font-weight: 700;">Team Analysis</span> • 
        <span style="color: #74C365; font-weight: 700;">Strategic Recommendations</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================
# Initialize chat history
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "🎯 **WELCOME TO THE COMMAND CENTER**\n\n"
                "Your AI analytics system is online and ready. "
                "I'm here to provide elite-level insights for your "
                "Fantasy Premier League strategy. "
                "What intelligence do you need today?"
            )
        }
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ===============================
# Chat Input (FIXED POSITION)
# ===============================
typed_input = st.chat_input(
    "⚡ Enter your query: players, fixtures, tactics, recommendations..."
)


# ===============================
# Quick Questions (INSIDE input)
# ===============================
if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

quick_questions = [
    "Top performing players this week",
    "Upcoming fixture analysis",
    "Best captain recommendation",
    "Team formation suggestions",
    "Injury updates"
]

st.markdown('<div class="fixed-question-icon">', unsafe_allow_html=True)
with st.popover("❓"):
    st.markdown("### Quick Questions")
    for i, q in enumerate(quick_questions):
        if st.button(q, key=f"qq_fixed_{i}"):
            st.session_state.quick_question = q
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)



# ===============================
# Prompt Resolution Logic
# ===============================
prompt = None

# Priority 1: quick question click
if st.session_state.quick_question:
    prompt = st.session_state.quick_question
    st.session_state.quick_question = None  # reset after use

# Priority 2: typed input (Enter)
elif typed_input:
    prompt = typed_input


# ===============================
# Chat Processing
# ===============================
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("⚡ ANALYZING DATA..."):
            try:
                # ---- YOUR EXISTING PIPELINE ----
                intent = classify_intent_llm(prompt)
                entities = extract_entities_with_llm(prompt)

                baseline_results = {}
                vector_results = []

                if retrieval_method == "Baseline Only":
                    graph_retriever = GraphRetrieval()
                    baseline_results = graph_retriever.retrieve_kg_context(
                        entities, intent
                    )
                    combined_context = combine_retrieval_results(
                        baseline_results=baseline_results
                    )

                elif retrieval_method == "Embedding Only":
                    vector_results = semantic_search(prompt, embed_model)
                    combined_context = combine_retrieval_results(
                        vector_results=vector_results
                    )

                elif retrieval_method == "Baseline + Embedding":
                    vector_results = answer_query(
                        prompt, entities, intent, embed_model
                    )
                    combined_context = combine_retrieval_results(
                        hybrid_results=vector_results
                    )

                context_str = (
                    "\n".join(map(str, combined_context))
                    or "No specific data found in the Knowledge Graph."
                )

                model_id = MODELS.get(model_choice)
                prompt_str = create_prompt_template(combined_context, prompt)

                full_response = query_llm(
                    model_id, prompt_str, OPENROUTER_API_KEY
                ).get("answer")

                message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = (
                    "⚠️ An error occurred during analysis. "
                    "Please retry your request."
                )
                st.error(f"⚠️ SYSTEM ERROR: {e}")

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
