# deployment/streamlit_app.py

import streamlit as st
import torch
from typing import Dict, List
import json
from datetime import datetime
import uuid

class ResearchAnalysisApp:
    """Streamlit application for research conversation analysis"""
    
    def __init__(self):
        self.load_model()
        self.initialize_session_state()
    
    def load_model(self):
        """Load the trained model"""
        self.model = torch.load('models/research_analysis_model.pt')
        self.model.eval()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'conversations' not in st.session_state:
            st.session_state.conversations = []
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
    
    def run(self):
        """Run the Streamlit application"""
        st.title("Research Conversation Analysis")
        
        # Sidebar for settings and history
        self.render_sidebar()
        
        # Main content area
        self.render_main_content()
        
        # Analysis panel
        self.render_analysis_panel()
    
    def render_sidebar(self):
        """Render the sidebar with settings and history"""
        with st.sidebar:
            st.header("Settings")
            
            # Analysis settings
            st.subheader("Analysis Configuration")
            st.slider("Trust Threshold", 0.0, 1.0, 0.7)
            st.slider("Bias Sensitivity", 0.0, 1.0, 0.5)
            
            # Conversation history
            st.subheader("Conversation History")
            for conv in st.session_state.conversations:
                if st.button(f"Conversation {conv['id'][:8]}"):
                    self.load_conversation(conv)
    
    def render_main_content(self):
        """Render the main conversation area"""
        st.header("Research Conversation")
        
        # Input area
        user_input = st.text_area("Enter your research discussion:")
        if st.button("Analyze"):
            self.analyze_input(user_input)
        
        # Current conversation display
        if st.session_state.conversations:
            current_conv = st.session_state.conversations[-1]
            self.display_conversation(current_conv)
    
    def render_analysis_panel(self):
        """Render the analysis results panel"""
        if st.session_state.current_analysis:
            st.header("Analysis Results")
            
            # Trust score
            trust_score = st.session_state.current_analysis['trust_score']
            st.metric("Trust Score", f"{trust_score:.2f}")
            
            # Authenticity metrics
            with st.expander("Authenticity Metrics"):
                self.display