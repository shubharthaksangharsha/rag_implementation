import os, tempfile
from pathlib import Path
from langchain.chains import ConversationChain
from langchain_community.callbacks import StreamlitCallbackHandler 
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_experimental.tools import PythonREPLTool

from langchain.agents import AgentType, initialize_agent, load_tools, Tool
import streamlit as st
from time import sleep 
import typer 
import psutil 
import sys 
import warnings


# Suppress specific warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Chatbot - Shubharthak")
st.title("Chat with Apsara")
memory = ConversationBufferMemory(k=5, return_messages=True)

def save_secrets():
    with open("./.streamlit/secrets.toml", "w") as file:
        file.write(f"groq_api_key = \"{st.session_state.groq_api_key}\"")


def generate_response(temperature, query, memory):
    if st.session_state.local:
        llm = Ollama(model='openchat', temperature=temperature)
        response = ConversationChain(llm=llm, memory=memory, return_final_only=True).invoke(query)
        print(response)
        response = response['response']
        st.session_state.messages.append((query, response))
        return response 
    
    llm = ChatGroq(api_key=st.session_state.groq_api_key, streaming=True
             , temperature=temperature)
    response = ConversationChain(llm=llm, memory=memory, return_final_only=True).invoke(query)
    response = response['response']
    st.session_state.messages.append((query, response))
    return response 


def input_fields():
    #
    
    with st.sidebar:
        if st.secrets.get('groq_api_key'):
            st.session_state.groq_api_key = st.secrets.groq_api_key
        else:
            st.session_state.groq_api_key = st.text_input("Groq API key", type="password")
            if st.session_state.groq_api_key:
                save_secrets()

    st.session_state.local =  st.toggle('Use Local Model: openchat')
    st.session_state.temperature = st.slider(label='Temperature', min_value=0.0, max_value=1.0, value=0.5)
    st.session_state.internet = st.checkbox('Use Agent', value=False)

def clear_history():
    st.session_state.messages = []
def boot():

    input_fields()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if st.session_state.local:
        llm = Ollama(model="openchat", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    else:
        llm = ChatGroq(api_key=st.session_state.groq_api_key, temperature=st.session_state.temperature, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), streaming=True)
    search = GoogleSerperAPIWrapper(serper_api_key=os.environ.get('SERPAPI_API_KEY'))
    tools = load_tools(['serpapi'], llm=llm)
    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
    )
    if query := st.chat_input():
        print(query)
        if 'exit' in query:
            st.stop()
        st.chat_message("human").write(query)
        if st.session_state.internet:
            with st.chat_message("ai"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent.run(query, callbacks=[st_callback])
                st.markdown(response)
        else:
            with st.spinner('Generating Response...'):            
                response = generate_response(temperature=st.session_state.temperature,
                                            query=query,
                                            memory=memory)
            st.chat_message("ai").markdown(response)
    st.button('Clear History', on_click=clear_history)
    

if __name__ == '__main__':
    boot()
    