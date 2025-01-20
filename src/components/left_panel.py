class ChatPanel:
    def render(self):
        st.markdown("### Chat")
        
        # Message history
        messages = st.session_state.get("messages", [])
        for msg in messages:
            self._render_message(msg)
        
        # Input
        if prompt := st.chat_input("Message..."):
            self._handle_message(prompt)
    
    def _render_message(self, msg):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    def _handle_message(self, prompt):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Get API response
        client = OpenAIClient()
        response = client.get_completion(st.session_state.messages)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })