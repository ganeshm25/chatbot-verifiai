import streamlit as st
import openai
from datetime import datetime
from typing import List, Dict, Optional
import os

# Page configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Styles
st.markdown("""
<style>
    /* Main layout */
    .stApp {
        background-color: #f9fafb;
    }
    
    .main-container {
        display: flex;
        height: 100vh;
    }
    
    /* Panels */
    .panel {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        height: calc(100vh - 80px);
        overflow-y: auto;
    }
    
    .left-panel {
        border-right: 1px solid #e5e7eb;
    }
    
    .chat-panel {
        margin: 0 1rem;
    }
    
    .right-panel {
        border-left: 1px solid #e5e7eb;
    }
    
    /* Messages */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .user-message {
        background-color: #f3f4f6;
    }
    
    .assistant-message {
        background-color: white;
        border: 1px solid #e5e7eb;
    }
    
    /* Input */
    .chat-input {
        position: fixed;
        bottom: 20px;
        left: 33%;
        width: 33%;
        padding: 1rem;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Topic buttons */
    .topic-button {
        width: 100%;
        text-align: left;
        padding: 0.5rem;
        margin: 0.25rem 0;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 0.25rem;
    }
    
    /* References */
    .reference-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'topics' not in st.session_state:
    st.session_state.topics = []
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = None
if 'references' not in st.session_state:
    st.session_state.references = []

class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def get_response(self, messages: List[Dict]) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            # Initialize placeholder for streaming
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            return full_response
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

class ResearchAssistant:
    def __init__(self):
        self.client = OpenAIClient(st.secrets["OPENAI_API_KEY"])
    
    def create_left_panel(self):
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)
        
        if st.button("+ New Topic"):
            topic = st.text_input("Enter topic name")
            if topic:
                if topic not in st.session_state.topics:
                    st.session_state.topics.append(topic)
                st.session_state.current_topic = topic
                st.session_state.messages = []
        
        st.markdown("### Topics")
        for topic in st.session_state.topics:
            if st.button(topic, key=f"topic_{topic}", help=f"Switch to {topic}"):
                st.session_state.current_topic = topic
                st.rerun()
    
    def create_chat_panel(self):
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        
        # Display current topic
        if st.session_state.current_topic:
            st.markdown(f"### {st.session_state.current_topic}")
        
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Message GPT-4..."):
            if not st.session_state.current_topic:
                st.warning("Please select or create a topic first.")
                return
            
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "topic": st.session_state.current_topic,
                "timestamp": datetime.now().isoformat()
            })
            
            # Get and add assistant response
            messages = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in st.session_state.messages]
            
            response = self.client.get_response(messages)
            if response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "topic": st.session_state.current_topic,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Extract references
                self.extract_references(response)
    
    def create_right_panel(self):
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        
        st.markdown("### References")
        if st.session_state.references:
            for ref in st.session_state.references:
                with st.expander(ref["title"]):
                    st.markdown(f"**Author:** {ref['author']}")
                    st.markdown(f"**Year:** {ref['year']}")
                    st.markdown(f"**Summary:** {ref['summary']}")
        else:
            st.info("No references available for this topic.")
    
    def extract_references(self, text: str):
        """Extract references from response text"""
        # Add reference extraction logic here
        pass
    
    def run(self):
        cols = st.columns([1, 2, 1])
        
        with cols[0]:
            self.create_left_panel()
        
        with cols[1]:
            self.create_chat_panel()
        
        with cols[2]:
            self.create_right_panel()

def main():
    assistant = ResearchAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
