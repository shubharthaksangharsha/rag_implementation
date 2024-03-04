import os, tempfile
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
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
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import PromptTemplate
import streamlit as st
from time import sleep 
import typer 
import psutil 
import sys 
import warnings
st.set_page_config(page_title="Home")
def boot():
    #
    st.header('Apsara AI', divider='rainbow')
    st.header('Powered :red[with] :blue[Advanced Capabilities] :sunglasses:')

    #
    st.page_link("pages/chatbot.py", label="Chatbot", icon='🤖', use_container_width=False,
                 help=''' Chat with Apsara: One of the Assistant who have real time knowledge also.
                 ''')
    st.page_link("pages/rag.py", label="RAG", icon='📚', 
                 help = ''' Chat with your PDFs/Docs/Websites with the assistant.
                 ''')

    #


if __name__ == '__main__':
    boot()
    