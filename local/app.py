from ingest import * 




if __name__ == '__main__':
    llm_input = input("Choose the LLM model\n1: openchat(local)\n2: groq-mixtral8x7b(internet): ")
    if llm_input == '1':
        llm = Ollama(model='openchat', temperature=0, callbacks=[StreamingStdOutCallbackHandler()])
    elif llm_input == '2':
        llm = ChatGroq(temperature=0, max_tokens=2048, model_name="mixtral-8x7b-32768", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])    

    # llm = Ollama(model='openhermes2.5-mistral', temperature=0, callbacks=[StreamingStdOutCallbackHandler()])
    
    # llm = ChatGroq(temperature=0, max_tokens=32768, model_name="mixtral-8x7b-32768")

    # docs = load_pdf_data('data/hiddenhindu.pdf')
    docs = load_pdf_data('data/resume.pdf')
    chunks = split_docs(docs)
    embedding_model = load_embedding_model(model_name="nomic-embed-text:latest")
    # embedding_model = load_embedding_model('huggingface')
    vectorstore = create_embeddings(chunks, embedding_model)
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
    # print(vectorstore.search(query='Who is Shubharthak', search_type='similarity'))
    
    # Load the stored db 
    # embed = load_embedding_model(model_name="nomic-embed-text:latest")
    # vectorstore = load_vector_store()
    # print(vectorstore.asearch(query='Who is Om shastri ? ', search_type='similarity'))
    # retriever = vectorstore.as_retriever()

    prompt = PromptTemplate.from_template(template=template)
    chat_history = []
    menu()
    while True:
        question = input('> ')
        if 'exit' in question:
            print('Thank you for using me...')
            break
        if 'chat_history' in question:
            print(chat_history)
            print('len:', len(chat_history))
            continue
        if 'clear_history' in question:
            chat_history = []
            continue
        if 'menu' in question:
            menu()
            continue
        chat_history.append('User: ' + question)
        chain = load_qa_chain(retriever=retriever, llm=llm, prompt=prompt)
        response = get_response(query=chat_history, chain=chain)
        chat_history.append('Response: ' + response)
        i += 1
        
        if i == buffer_size:
            print('Clearing...')
            chat_history = chat_history[buffer_size:]
            i = 0
            