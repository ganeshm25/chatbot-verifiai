import streamlit as st
from src.components.left_panel import LeftPanel
from src.components.chat_panel import ChatPanel
from src.components.reference_panel import ReferencePanel
import openai
from datetime import datetime
from typing import List, Dict


# Configure page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Research Assistant"
)

st.markdown("""
<style>
    /* Reset */
    .stApp {
        background-color: #f9fafb;
    }
    
    /* Panel Layout */
    .left-panel {
        background: white;
        border-right: 1px solid #e5e7eb;
        height: 100vh;
        padding: 1rem;
    }
    
    .chat-panel {
        background: white;
        padding: 2rem;
        height: calc(100vh - 80px);
        overflow-y: auto;
    }
    
    .reference-panel {
        background: white;
        border-left: 1px solid #e5e7eb;
        height: 100vh;
        padding: 1rem;
    }
    
    /* Messages */
    .stChatMessage {
        background: transparent;
        border: none;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .stChatMessage.user {
        background: #f3f4f6;
    }
    
    .stChatMessage.assistant {
        background: white;
        border: 1px solid #e5e7eb;
    }
    
    /* Input */
    .stChatInput {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-top: 1rem;
    }
    
    /* Buttons */
    .stButton button {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        color: #374151;
        width: 100%;
        text-align: left;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize panels
left_panel = LeftPanel()
chat_panel = ChatPanel()
reference_panel = ReferencePanel()

# Create layout
cols = st.columns([1, 2, 1])
with cols[0]: left_panel.render()
with cols[1]: chat_panel.render()
with cols[2]: reference_panel.render()


class GPTClient:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def get_response(self, messages: List[Dict]) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

class ResearchSession:
    def __init__(self):
        self.messages = []
        self.references = []
        self.current_topic = None

if 'session' not in st.session_state:
    st.session_state.session = ResearchSession()

def main():
    left_col, main_col, right_col = st.columns([1, 2, 1])
    
    with left_col:
        st.markdown("### Research Topics")
        create_left_sidebar()
    
    with main_col:
        st.markdown("### Research Assistant")
        create_main_chat()
    
    with right_col:
        st.markdown("### References")
        create_right_panel()

def create_left_sidebar():
    if st.button("New Topic"):
        topic = st.text_input("Topic name:")
        if topic:
            st.session_state.session.current_topic = topic
            st.session_state.session.messages = []
    
    st.markdown("#### Previous Topics")
    topics = list(set([msg.get("topic") for msg in st.session_state.session.messages if msg.get("topic")]))
    for topic in topics:
        if st.button(topic, key=f"topic_{topic}"):
            st.session_state.session.current_topic = topic

def create_main_chat():

    # Container for chat history
    chat_container = st.container()
    chat_container.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    
    # Messages area with scrolling
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Fixed input at bottom
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    if prompt := st.chat_input("Message..."):
        handle_message(prompt)
    st.markdown('</div>', unsafe_allow_html=True)

    # for message in st.session_state.session.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])
    
    # if prompt := st.chat_input("Enter your research query..."):
    #     if not st.session_state.session.current_topic:
    #         st.warning("Please select or create a new topic first.")
    #         return
        
    #     st.session_state.session.messages.append({
    #         "role": "user",
    #         "content": prompt,
    #         "topic": st.session_state.session.current_topic,
    #         "timestamp": datetime.now().isoformat()
    #     })
        
    #     client = GPTClient(st.secrets["OPENAI_API_KEY"])
    #     messages = [{"role": msg["role"], "content": msg["content"]} 
    #                for msg in st.session_state.session.messages]
        
    #     response = client.get_response(messages)
    #     if response:
    #         st.session_state.session.messages.append({
    #             "role": "assistant",
    #             "content": response,
    #             "topic": st.session_state.session.current_topic,
    #             "timestamp": datetime.now().isoformat()
    #         })
            
    #         extract_references(response)

def create_right_panel():
    if st.session_state.session.references:
        for ref in st.session_state.session.references:
            with st.expander(ref["title"]):
                st.markdown(f"**Author:** {ref['author']}")
                st.markdown(f"**Year:** {ref['year']}")
                st.markdown(f"**Summary:** {ref['summary']}")

def extract_references(text: str) -> List[Dict]:
    references = []
    # Add reference extraction logic
    return references

if __name__ == "__main__":
    main()