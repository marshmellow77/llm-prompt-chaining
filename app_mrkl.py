import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "3098db6b722525d8782bff2e879830e73c50ea50b6fe1338729457c355a56a36"

st.set_page_config(
    page_title="Enhanced LLM with integrated search capability",
    page_icon=":robot:"
)

st.header("Enhanced LLM with integrated search capability")

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi"], llm=llm)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = agent.run(user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
