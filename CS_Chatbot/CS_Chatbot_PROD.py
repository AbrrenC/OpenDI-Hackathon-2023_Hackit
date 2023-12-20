import streamlit as st
import requests
import re

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

st.set_page_config(page_title="🛍️ Online Shopping Platform Customer Service Chatbot 🤖")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
st.sidebar.title('🛍️ Online Shopping Platform Customer Service Chatbot 🤖')
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

prompt = st.text_input("How may I help you today 😊") if st.session_state.get("messages") is not None else ""

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            response = query(
                {"inputs": f"{string_dialogue} {prompt} Assistant:",
                "parameters": {"max_new_tokens": 500, "return_full_text": False},
                })
            cleaned_response = []
            for item in response:
                if "generated_text" in item:
                    cleaned_response.append({item["generated_text"]})  # Extract only the generated text
                else:
                    cleaned_response.append(item)  # Keep other parts of the response intact
            for item in cleaned_response:
                st.markdown(item)
            st.session_state.messages.append({"role": "assistant", "content": cleaned_response})