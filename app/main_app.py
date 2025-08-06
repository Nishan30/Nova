# app/main_app.py

import streamlit as st
import os
from rag_backend import get_journal_context, process_documents, get_qa_chain
import streamlit as st
import os
import random  # Add this import
from rag_backend import process_documents, get_qa_chain, init_database, save_journal_entry

# Define the path to the database directory to check for existence
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_DIR = os.path.join(PROJECT_ROOT, 'db')
chat_tab, journal_tab = st.tabs(["💬 Chat", "✍️ Journal"])

st.set_page_config(page_title="Enclave - Your Offline Companion", layout="wide")
st.title("Enclave 💬")
st.markdown("Your private, offline companion for memory and conversation.")
init_database()

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Controls")
    st.markdown("Place your `.pdf` or `.txt` files in the `docs` folder.")
    if st.button("Process New Documents", type="primary"):
        with st.spinner("Processing documents... This may take a moment."):
            docs_processed = process_documents()
            if docs_processed > 0:
                st.success(f"Successfully processed {docs_processed} new documents!")
                st.session_state.qa_chain = None 
            else:
                st.info("No new documents were found to process.")
    st.markdown("---")
    st.info("Everything runs 100% locally on your machine. Your data never leaves your computer.")

if st.session_state.qa_chain is None and os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    with st.spinner("Loading memory..."):
        st.session_state.qa_chain = get_qa_chain()
        st.toast("Memory loaded successfully!", icon="🧠")

st.header("Chat with your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.qa_chain is None:
            st.warning("The memory is not loaded. Please process your documents first.")
            st.stop()
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(prompt)
            answer = response.get('result', 'Sorry, I encountered an error.')
            st.markdown(answer)
            with st.expander("Show Sources"):
                for doc in response.get('source_documents', []):
                    source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    st.write(f"**Source:** `{source_name}`")
                    st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

with journal_tab:
    st.header("My Daily Journal")
    st.write("Reflect on your day, your thoughts, and your feelings.")
    
    # "Amazing" Touch: Add random prompts to inspire the user
    journal_prompts = [
        "What was a small win for you today?",
        "What's been on your mind lately?",
        "Describe a challenge you faced and how you handled it.",
        "What are you grateful for right now?",
        "What are you looking forward to this week?"
    ]
    st.info(f"**Prompt:** {random.choice(journal_prompts)}")
    
    # Text area for the journal entry
    journal_entry = st.text_area("Write your thoughts here...", height=300, label_visibility="collapsed")
    
    # Save button logic
    if st.button("Save Entry", type="primary"):
        if journal_entry.strip():
            save_journal_entry(journal_entry)
            st.success("Your journal entry has been saved to your local memory.")
        else:
            st.warning("Please write something before saving.")


# --- Chat Tab UI ---
with chat_tab:
    st.header("Chat with your memories")

    # This part is your existing chat interface code - move it inside this 'with' block
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your documents or your journal..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.qa_chain is None:
                st.warning("The memory is not loaded. Please process your documents first.")
                st.stop()
            with st.spinner("Thinking..."):
                # We will upgrade this response logic in the next step
                journal_context = get_journal_context(prompt)
            
                final_prompt = prompt
                if journal_context:
                    # 2. If context is found, create an "augmented prompt"
                    st.info("Searching journal entries for emotional context...")
                    final_prompt = (
                        f"Based on my recent journal entries provided below, please answer my question. "
                        f"Analyze the tone and content of the entries to inform your response.\n\n"
                        f"--- Journal Context ---\n{journal_context}\n\n"
                        f"--- My Question ---\n{prompt}"
                    )
                
                # 3. Pass the final_prompt (either original or augmented) to the chain
                response = st.session_state.qa_chain.invoke(final_prompt)
                answer = response.get('result', 'Sorry, I encountered an error.')
                st.markdown(answer)
                
                with st.expander("Show Sources"):
                    for doc in response.get('source_documents', []):
                        source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        st.write(f"**Source:** `{source_name}`")
                        st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})