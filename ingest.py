# Importing libraries 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler, StreamingStdOutCallbackHandler
import textwrap
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq




# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
# readonlymemory = ReadOnlySharedMemory(memory=memory)

#Load PDFs file 
def load_pdf_data(file_path):
    # loader = PyPDFDirectoryLoader(path=file_path, glob='*.pdf', extract_images=True)
    # print(file_path)
    loader = PyMuPDFLoader(file_path=file_path)

    docs = loader.load()

    return docs 


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

def load_embedding_model(model_name):
    embedding = OllamaEmbeddings(model='nomic-embed-text:latest')
    return embedding

def load_vector_store(stored_path="vectorstore", embeddings=load_embedding_model(model_name="nomic-embed-text:latest")):
    vectorstore = FAISS.load_local(stored_path, embeddings=embeddings)
    return vectorstore


def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    vectorstore.save_local(storing_path)

    return vectorstore

# Loading the retriever
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt} # customizing the prompt
    )
# Prettifying the response
def get_response(query, chain):
    # Getting response from chain
    response = chain.invoke({'query': query})
    # # Wrapping the text for better output in Jupyter Notebook
    # wrapped_text = textwrap.fill(response['result'], width=100)
    # print(wrapped_text)
    print('')
    return response['result']

# Menu function
def menu():
    print('Loading...')
    print('Welcome to the chat!')
    print('Type chat_history to see the chat history.')
    print('Type exit to quit.')

# Global variables
i, buffer_size = 0, 5

# Formatting the prompt
template = """
    ### System:
    You are an respectful and honest assistant. You have to answer the user's 
    questions using only the context provided to you. If you don't know the answer, 
    just say you don't know. Don't try to make up an answer. 
    Chat History is also provided as list of messages(chat_history + question) to answer the question in more better way.

    
    ### Context:
    {context}

    
    ### User:
    {question}

    ### Response:
"""
template2 = """
    ### System: 
    You are system model which answer query user input based on the context I will provide. It should output the question user input in an stricly formatted JSON format.
    The resume will be passed. You should understand the resume well and answer the question user gives in stricly JSON format.  Keep in mind about the following:
    * Years of experience in a particular field or industry
    * Your output should be in JSON format only.
    * JSON format should contain the 'question': question_asked,'answer': answer 
    * If no information is provided use default value as 0 for experience, and use default value as No for choices. 

Based on the information contained in the resume, you have to answer the question user input and answers in the JSON format described above. Please ensure that the questions are relevant, specific, and unambiguous.

    ### Context:
    {context}

    ### User:
    {question}
"""
if __name__ == '__main__':
    pass
    # # docs = load_pdf_data('data/hiddenhindu.pdf')
    # docs = load_pdf_data('./data/resume.pdf')
    # chunks = split_docs(docs)
    # embedding_model = load_embedding_model(model_name="nomic-embed-text:latest")
    # vectorstore = create_embeddings(chunks, embedding_model)
    # retriever = vectorstore.as_retriever()
    # prompt = PromptTemplate.from_template(template=template)
    # llm = Ollama(model='openchat', temperature=0, callbacks=[StreamingStdOutCallbackHandler()])
    # chain = load_qa_chain(retriever, llm, prompt)
    # while True:
    #     question = input('Enter your question: ')
    #     if question == 'exit':
    #         break
    #     response = get_response(question, chain)
        