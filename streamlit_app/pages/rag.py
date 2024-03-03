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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_dirs(): 
        # Check if "data" folder exists
        print('Checking data folder exists or not ')
        if not os.path.exists("data"):
            # Create "data" folder
            os.makedirs("data")
            print("Created 'data' folder")

            # Create "vector_store" and "tmp" folders inside "data" folder
            os.makedirs(os.path.join("data", "vector_store"))
            os.makedirs(os.path.join("data", "tmp"))
            print("Created 'vector_store' and 'tmp' folders inside 'data' folder")
        else:
            print("'data' folder already exists")
        #check if ".streamlit" folder exists
        if not os.path.exists('.streamlit'):
            os.makedirs('.streamlit')
            print('Created .Streamlit folder')
        else:
            print('.streamlit folder already exists')

        secrets_file = os.path.join('.streamlit', 'secrets.toml')
        if not os.path.exists(secrets_file):
            # Create secrets.toml file
            os.system(f'touch {secrets_file}')
            print(f"File '{secrets_file}' created successfully.")
        else:
            print(f"File '{secrets_file}' already exists.")


create_dirs()
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="Rag - Shubharthak")
st.title("Chat with Docs/Website")


def save_secrets():
    with open("./.streamlit/secrets.toml", "w") as file:
        file.write(f"groq_api_key = \"{st.session_state.groq_api_key}\"")

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts, save=False):
    
    if st.session_state.hugging:
        hf = HuggingFaceEmbeddings
        vectordb = Chroma.from_documents(texts, embedding=HuggingFaceEmbeddings(model_kwargs={'device': 'cpu'}),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    else:
        vectordb = Chroma.from_documents(texts, embedding=OllamaEmbeddings(model='nomic-embed-text'),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    if st.session_state.save_local:
        vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def query_llm(retriever, query):
    if st.session_state.local:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=Ollama(model='openchat', temperature=0.0),
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=False,
        )    
        response =  qa_chain.invoke({'question': query, 'chat_history': st.session_state.messages})
        response = response['answer']
        st.session_state.messages.append((query, response))
        return response 
    
    qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGroq(api_key=st.session_state.groq_api_key, streaming=True),
            retriever=retriever,
            return_source_documents=True,
    )
    response =  qa_chain.invoke({'question': query, 'chat_history': st.session_state.messages})
    response = response['answer']
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
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    st.session_state.hugging =  st.toggle('HuggingFaceEmbeddings')
    st.session_state.save_local = st.toggle('Save Docs Locally')

def process_documents():
    if not st.session_state.groq_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                print('loading...')
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    print('inside tmp 1')
                    print(tmp_file)
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                print('loaded docs')
                #
                for _file in TMP_DIR.iterdir():
                    print('inside tmp 2')
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                print('Done splitting the text')
                # print(texts)
                #
                if st.session_state.source_docs:
                    print('in source')
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                    st.success(body='Successfully uploaded the data, Now you can chat...')
        except Exception as e:
            st.error(f"An error occurred: {e}")
def boot():
    

    input_fields()
    #
    st.button("Submit Documents", on_click=process_documents)
    
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        if 'exit' in query:
            st.stop()
        st.chat_message("human").write(query)
        with st.spinner('Generating Response...'):            
            response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").markdown(response)
        

if __name__ == '__main__':
    boot()
    