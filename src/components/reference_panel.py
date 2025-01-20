class ReferencePanel:
    def render(self):
        st.markdown("### References")
        
        refs = st.session_state.get("references", [])
        for ref in refs:
            with st.expander(ref["title"]):
                st.markdown(f"**Author:** {ref['author']}")
                st.markdown(f"**Year:** {ref['year']}")
                st.markdown(f"**Summary:** {ref['summary']}")