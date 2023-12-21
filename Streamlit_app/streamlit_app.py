# import libraries
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from api_links import lakera_guard_endpoints, lakera_guard_api_key, API_URL_Llama2

headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL_Llama2, headers=headers, json=payload)
    return response.json()

# define the make post request from Lakera API
def make_post_request(url, prompt, lakera_guard_api_key):
    response = requests.post(
        url,
        json={"input": prompt},
        headers={"Authorization": f"Bearer {lakera_guard_api_key}"}
    )
    return response.json()

# define the prompt guard function
def prompt_guard(endpoints, prompt, lakera_guard_api_key):
    with ThreadPoolExecutor() as executor:
        results = {}
        futures = {executor.submit(make_post_request, url, prompt, lakera_guard_api_key): endpoint for endpoint, url in endpoints.items()}

        for future in futures:
            endpoint = futures[future]
            response = future.result()
            results[endpoint] = response

    messages = {
        "moderation": "For assistance, kindly use respectful language. How can I help you with your inquiry?",
        "prompt_injection": "Sorry, I cannot answer this answer. Please provide another inquiry.",
        "pii": "For information safety, please do not provide any personal details.",
        "sentiment": "For assistance, kindly use respectful language. How can I help you with your inquiry?",
        "unknown_links": "Sorry, I cannot access any links via the chat.",
        "relevant_language": "Sorry, we do not support any other languages except English, please describe your inquiry in English."
    }

    for endpoint, result in results.items():
        if "results" in result and len(result["results"]) > 0 and "flagged" in result["results"][0] and result["results"][0]["flagged"]:
            return messages[endpoint]

st.set_page_config(page_title="🛍️ Online Shopping Platform Customer Service Chatbot 🤖")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
st.sidebar.title('🛍️ Online Shopping Platform Customer Service Chatbot 🤖')
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

prompt = st.text_input("How may I help you today 😊") if st.session_state.get("messages") is not None else ""

guard_output = prompt_guard(lakera_guard_endpoints, prompt, lakera_guard_api_key)

if guard_output is None:  # Check if the prompt is safe
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.spinner("Thinking..."):
                string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
                response = query({
                    "inputs": f"{string_dialogue} {prompt} Assistant:",
                    "parameters": {"max_new_tokens": 500, "return_full_text": False},
                    "temperature": 1,
                })
                cleaned_response = []
                for item in response:
                    if "generated_text" in item:
                        cleaned_response.append(item["generated_text"])  # Extract only the generated text
                for item in cleaned_response:
                    st.markdown(item)
                st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
else:
    st.session_state.messages.append({"role": "assistant", "content": guard_output})
    st.write(guard_output)



