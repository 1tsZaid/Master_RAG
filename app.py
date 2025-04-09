import streamlit as st
from master_rag import request, refresh_chain
from upsert import upsert

INDEX_NAME = "master-rag"

tab1, tab2, tab3 = st.tabs(["Chatbot", "Upload to Store", "New RAG"])

with tab1:
    with st.container():
        st.title("Master Chatbot")
        st.caption("Assist using different RAG systems")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Rerun to display the user message immediately
        st.rerun()

    # This part will only run after the rerun if a new message was added
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_message = st.session_state.messages[-1]["content"]
        
        # Get response from the RAG system
        assistant_response = request(user_message)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Rerun again to display the assistant message
        st.rerun()

with tab2:
    uploaded_files = st.file_uploader("Choose a new document", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        upsert(INDEX_NAME, uploaded_file.getvalue().decode("utf-8"))

with tab3:
    st.write("Add a new RAG system")
    with st.form("my_form"):
        name = st.text_input("Name", "medical")
        description = st.text_area("Description", "Good for answering questions about medical conditions")
        prompt = st.text_area("System prompt", "You are an assistant for diagnosing and treating medical conditions")

        submitted = st.form_submit_button("Submit")
        if submitted:
            refresh_chain(name, description, prompt)
            st.write("Submitted!")
