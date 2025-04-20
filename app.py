import streamlit as st
import json

from master_rag import request, refresh_chain
from upsert import upsert
from config import INDEX_NAME

tab1, tab2, tab3 = st.tabs(["Chatbot", "Upload to Store", "New RAG"])

with tab1:
    with st.container():
        st.title("Master Chatbot")
        st.caption("Assist using different RAG systems")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        # Display existing messages
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        
        # Check if we need to show the "Thinking..." indicator
        if st.session_state.get("thinking", False):
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    user_message = st.session_state.messages[-1]["content"]

                    assistant_response = request(user_message)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    
                    # Clear the thinking flag
                    st.session_state.thinking = False
                    
                    # Rerun again to display the assistant message and remove spinner
                    st.rerun()
        
    if prompt := st.chat_input("Your message"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Set thinking flag to true
        st.session_state.thinking = True
        # Rerun to display the user message and thinking indicator
        st.rerun()


with tab2:
    uploaded_files = st.file_uploader("Choose a new document", accept_multiple_files=True, key="uploader")

if uploaded_files and not st.session_state.get("uploaded"):
    with st.spinner("Uploading..."):
        for uploaded_file in uploaded_files:
            upsert(INDEX_NAME, uploaded_file.getvalue().decode("utf-8"))
    st.session_state.uploaded = True
    st.success("Done uploading!")
elif not uploaded_files:
    st.session_state.uploaded = False

with tab3:
    st.write("Add a new RAG system")
    
    with st.form("my_form"):
        name = st.text_input("Name", "medical")
        description = st.text_area("Description", "Good for answering questions about medical conditions")
        prompt = st.text_area("System prompt", "You are an assistant for diagnosing and treating medical conditions")

        submitted = st.form_submit_button("Submit")
        message = None

        if submitted:
            submit = refresh_chain(name, description, prompt)
            message = "New RAG system added" if submit else "RAG system already exists"
        
        if message:
            st.write(message)

    prompt_infos = []
    prompt_file = "prompt_templates.txt"
    try:
        with open(prompt_file, "r") as f:
            prompt_infos = json.load(f)

            st.table({"RAG Name": [rag["name"] for rag in prompt_infos], 
                        "Description": [rag["description"] for rag in prompt_infos] })
    except json.JSONDecodeError:
        pass
