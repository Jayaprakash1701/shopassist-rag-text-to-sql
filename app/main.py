import streamlit as st
from router import router
from faq import ingest_faq_data, faq_chain
from pathlib import Path
from sql import sql_chain

faq_path = Path(__file__).parent / "resources" / "faq.csv"

ingest_faq_data(faq_path)

def ask(query):
    route_choice = router(query)
    route_name = route_choice.name if route_choice is not None else None

    if route_name == "faq":
        answer = faq_chain(query)
        return answer
    else:
        answer = sql_chain(query)
        return answer

st.title("E-commerce Assistant")

query = st.chat_input("Ask me anything about our products or services!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query:
    with st.chat_message("user"):
        st.markdown(query)

    response = ask(query)
    with st.chat_message("assistant"): 
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})    
    

