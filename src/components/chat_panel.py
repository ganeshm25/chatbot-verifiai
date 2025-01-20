class LeftPanel:
    def render(self):
        st.markdown("### Research Topics")
        
        # New topic button
        if st.button("+ New Topic"):
            with st.form("new_topic"):
                topic = st.text_input("Topic name")
                if st.form_submit_button("Create"):
                    st.session_state.current_topic = topic
        
        # Topic list
        st.markdown("#### Previous Topics")
        for topic in st.session_state.get("topics", []):
            if st.button(topic, key=f"topic_{topic}"):
                st.session_state.current_topic = topic