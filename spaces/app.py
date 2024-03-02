
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain_community.document_loaders import PyMuPDFLoader


@cl.on_chat_start
async def on_chat_start():
    files = None #Initialize variable to store uploaded files

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,# Optionally limit the file size
            timeout=180, # Set a timeout for user response,
        ).send()

    file = files[0] # Get the first uploaded file
    print(file) # Print the file object for debugging
    
     # Sending an image with the local file path
    elements = [

    ]
    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...",elements=elements)
    await msg.send()

    # Read the PDF file
    pdf = PyMuPDFLoader(file_path=file.path)
    docs = pdf.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents=docs)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(chunks))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(FAISS.from_documents(chunks, embeddings=embeddings))
    docsearch.save_local("vectorstore")
    
    

    # Create a chain that uses the Chroma vector store
    chain = RetrievalQA.from_chain_type(
        Ollama(model="openchat", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()]),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()
    #store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
     # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 
    #call backs happens asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] # Initialize list to store text elements
    
    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
         # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    #return results
    await cl.Message(content=answer, elements=text_elements).send()