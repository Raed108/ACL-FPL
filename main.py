import streamlit as st
import pandas as pd
import os
import time
from dotenv import load_dotenv
from google import genai

# --- CRITICAL: LOAD ENV VARS BEFORE IMPORTING YOUR MODULES ---
load_dotenv() 




# --- Import User Modules ---
# Ensure your folder structure contains __init__.py files so these imports work
from InputPreprocessing.intent_classifier import classify_intent
from InputPreprocessing.entity_extractions import extract_entities, extract_entities_with_llm
from GraphRetrievalLayer.Baseline import GraphRetrieval
from LLMLayer.Baseline_Embeddings_Combined import combine_retrieval_results
# Note: Assuming you have an embeddings file. If not, we use a placeholder below.
# from GraphRetrievalayer.embeddings import VectorRetrieval 

from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# --- Page Configuration ---
st.set_page_config(
    page_title="FPL Graph-RAG Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Cute & Professional" Look ---
st.markdown("""
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #000000;
    }
    
    /* Chat Message Bubbles */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* FPL Theme Colors */
    h1 {
        color: #37003c; /* FPL Dark Purple */
        font-family: 'Arial', sans-serif;
        font-weight: 800;
    }
    h2, h3 {
        color: #00ff85; /* FPL Green */
        background-color: #37003c;
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #37003c;
        color: white;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #00ff85 !important; /* Green Text */
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #00ff85;
        color: #37003c;
        border-radius: 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #02db72;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def generate_llm_response(model_name, context, user_query, intent):
    """
    Generates the final answer using the selected LLM.
    """
    # Construct the System Prompt
    system_prompt = f"""
    You are an expert Fantasy Premier League (FPL) Assistant.
    
    **Your Task:** Answer the user's question based strictly on the provided Context.
    **User Intent:** {intent}
    
    **Context from Knowledge Graph:**
    {context}
    
    **Guidelines:**
    1. Be helpful, concise, and enthusiastic (like a football commentator).
    2. If the answer is not in the context, admit you don't know based on current data.
    3. Use bolding for player names and key stats.
    4. If recommending players, explain WHY based on the stats provided.
    """

    try:
        if "gemini" in model_name.lower():
            response = client.models.generate_content(
                model=model_name,
                contents=f"User Query: {user_query}\n\nSystem Instructions: {system_prompt}"
            )
            return response.candidates[0].content.parts[0].text
        else:
            # Placeholder for other models (GPT/Claude) if you add them later
            return "⚠️ Model not connected. Using Mock Response."
    except Exception as e:
        return f"❌ Error communicating with LLM: {str(e)}"

def format_context_for_display(results):
    """
    Formats the raw dictionary results into a readable string/JSON for the UI.
    """
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

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/56/Premier_League_Logo.svg/1200px-Premier_League_Logo.svg.png", width=100)
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    
    # 1. Model Selection (Requirement 3.c & 4.c)
    model_choice = st.selectbox(
        "🧠 Select LLM Model",
        ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b"],
        index=0
    )
    
    # 2. Retrieval Method (Requirement 4.c)
    retrieval_method = st.radio(
        "🔍 Retrieval Strategy",
        ["Baseline (Cypher Only)", "Embeddings (Vector Only)", "Hybrid (Best Results)"],
        index=0
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Hybrid retrieval usually provides the most accurate results by combining exact matches with semantic search.")

# --- Main Interface ---
st.title("⚽ FPL Graph-RAG Assistant")
st.markdown("Ask me about **Player Stats**, **Fixtures**, **Team Analysis**, or get **Fantasy Recommendations**!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello manager! 👋 Who are we analyzing today?"}
    ]

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Processing Pipeline ---
if prompt := st.chat_input("Ex: How many goals did Haaland score in 2023?"):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Processing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🔍 Scouting the database..."):
            try:
                # --- Step A: Preprocessing ---
                # 1. Intent Classification
                intent = classify_intent(prompt)
                
                # 2. Entity Extraction (Using your hybrid function)
                entities = extract_entities(prompt)
                
                # --- Step B: Retrieval ---
                baseline_results = {}
                vector_results = []
                
                # Baseline Retrieval
                if retrieval_method in ["Baseline (Cypher Only)", "Hybrid (Best Results)"]:
                    graph_retriever = GraphRetrieval()
                    baseline_results = graph_retriever.retrieve_kg_context(entities, intent)
                    # graph_retriever.close()
                
                # Vector Retrieval (Placeholder - Connect your Embedding file here)
                if retrieval_method in ["Embeddings (Vector Only)", "Hybrid (Best Results)"]:
                    # vector_retriever = VectorRetrieval() 
                    # vector_results = vector_retriever.search(prompt)
                    vector_results = [] # Leaving empty as per your provided code snippets
                
                # --- Step C: Combination ---
                # Flatten baseline results for the combiner
                flattened_baseline = format_context_for_display(baseline_results)
                
                # Combine (Your function)
                combined_context = combine_retrieval_results(flattened_baseline, vector_results)
                
                # Stringify context for LLM
                context_str = "\n".join([str(item) for item in combined_context])
                
                if not context_str:
                    context_str = "No specific data found in the Knowledge Graph for this query."

                # --- Step D: LLM Generation ---
                full_response = generate_llm_response(model_choice, context_str, prompt, intent)
                
                message_placeholder.markdown(full_response)
                
                # --- Step E: Transparency / Debug UI (Requirement 4.a & 4.b) ---
                # This section makes it "Professional" and meets the "View KG Context" requirement
                with st.expander("🧐 View Retrieval Details (Under the Hood)"):
                    tab1, tab2, tab3 = st.tabs(["Entities & Intent", "Raw KG Context", "Cypher Results"])
                    
                    with tab1:
                        st.subheader("Detected Intent")
                        st.code(intent)
                        st.subheader("Extracted Entities")
                        st.json(entities)
                    
                    with tab2:
                        st.subheader("Context passed to LLM")
                        st.text_area("Raw Context", context_str, height=200)
                        
                    with tab3:
                        if baseline_results:
                            st.subheader("Structured Data (Cypher)")
                            # Convert dict results to dataframe for pretty table
                            for query_key, data in baseline_results.items():
                                st.markdown(f"**Query: {query_key}**")
                                if data:
                                    st.dataframe(pd.DataFrame(data))
                                else:
                                    st.warning("No results for this specific query.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = "I encountered an error processing your request."

    # 3. Update History
    st.session_state.messages.append({"role": "assistant", "content": full_response})

