import os, tempfile
from pathlib import Path
from langchain.chains import ConversationChain, LLMChain
from langchain_community.callbacks import StreamlitCallbackHandler 
from langchain_community.chat_models import ChatOllama
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

#agents modules
from langchain.agents import AgentType, initialize_agent, load_tools, get_all_tool_names


from mytools import *
import streamlit as st
from time import sleep 
import typer 
import psutil 
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Default"

st.set_page_config(page_title="Apsara Chat")
st.title("Chat with Apsara")
memory = ConversationBufferMemory(k=5, return_messages=True)

def save_secrets():
    with open("./.streamlit/secrets.toml", "w") as file:
        file.write(f"groq_api_key = \"{st.session_state.groq_api_key}\"")


def extract_last_human_message(conversation):
    last_human_message = None

    for message in reversed(conversation):
        if message.startswith("Human:"):
            last_human_message = message
            break

    if last_human_message:
        return last_human_message.split(": ", 1)[1]
    else:
        return None

def generate_response(temperature, query, cb):
    if st.session_state.local:
        llm = ChatOllama(model='openchat', temperature=temperature)
    else:
        llm = ChatGroq(api_key=st.session_state.groq_api_key, streaming=True, temperature=temperature)
    response = ConversationChain(llm=llm, return_final_only=True).invoke(query)
    response = response['response']
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

    # st.session_state.max_tokens = st.selectbox(options=['512', '4096', '32768'],label='Max Tokens', help='uses max output tokens to set for ChatGroq')
    st.session_state.temperature = st.slider(label='Temperature', min_value=0.0, max_value=1.0, value=0.5, help='Define the random answers for LLM to generate, the higher the temperature more creative models get in output answers')
    st.session_state.local =  st.checkbox('Use Local LLM', help="Use local LLM Ollama openchat")
    st.session_state.internet = st.checkbox('Use Agent', help='Turn on Advance Capablities (for instance: Real time Data)')

def clear_history():
    st.session_state.messages = []
    st.session_state.history = []
def boot():
    if "history" not in st.session_state:
        st.session_state.history = []

    input_fields()
    if "prompt" not in st.session_state:
        st.session_state.prompt = PromptTemplate(input_variables=["question"], template="""
        The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context. 
        If the AI does not know the answer to a question, it truthfully says it does not know.
        Messages also contains the history of previous conversation(in a List format) for easy answering. 
                                                 
        Human:{question}
        AI:""")
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if st.session_state.local:
        llm = ChatOllama(model="openchat", temperature=st.session_state.temperature, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    else:
        llm = ChatGroq(api_key=st.session_state.groq_api_key, temperature=st.session_state.temperature, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), streaming=True)
    
    print(get_all_tool_names())
    agents = ["zero-shot-react-description", "conversational-react-description",
           "chat-zero-shot-react-description", "chat-conversational-react-description", 
           "structured-chat-zero-shot-react-description"]
    tools = load_tools(
        ["serpapi", "llm-math"], 
        llm=llm)
    tools.append(mylocation)
    tools.append(read_tool)
    tools.append(write_tool)
    tools.append(weather_tool)
    tools.append(python_tool)
    tools.append(get_today_date)
    agent = initialize_agent(   
        tools=tools, llm=llm, 
        agent=AgentType(agents[-1]), 
        verbose=True, 
        handle_parsing_errors=True)
    
    
    
    if query := st.chat_input():
        if 'exit' in query:
            st.stop()
        st.chat_message("human").write(query)
        st.session_state.history.append(f'Human: {query}')
        if st.session_state.internet:
            with st.chat_message("ai"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent.run(query, callbacks=[st_callback])
                st.markdown(response)
                st.session_state.messages.append((query, response))
        else:
            with st.chat_message("ai"):
                with st.spinner('Generating Response...'):            
                    st_callback = StreamlitCallbackHandler(st.container())
                    response = generate_response(temperature=st.session_state.temperature,
                                                query=st.session_state.history, cb=st_callback)
                    print(response)
                    st.markdown(response)
                    st.session_state.history.append(f'AI: {response}')
                    st.session_state.messages.append((query, response))
    st.button('Clear History', on_click=clear_history)
    print(len(st.session_state.history))
    

if __name__ == '__main__':
    boot()
    