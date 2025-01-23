from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

os.environ["HUGGINGFACE_TOKEN"]=st.secrets["token"]

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-l6-v2")

# Streamlit
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDF and chat with their content")

api_key=st.secrets["api"]

if api_key:
    llm=ChatGroq(api_key=api_key,model="Gemma2-9b-It")
    
    session_id=st.text_input("Session ID",value="21231321")
    
    if 'store' not in st.session_state:
        st.session_state.store={}
        
    uploaded_file=st.file_uploader("Chose a pdf file to upload",type="pdf",accept_multiple_files=True)
    
    ## Processing uploaded files
    if uploaded_file:
        documents=[]
        for doc in uploaded_file:
            tempPdf=f"./temp.pdf"
            with open(tempPdf,"wb") as file:
                file.write(doc.getvalue())
                file_name=doc.name
            
            loader=PyPDFLoader(tempPdf)
            docs=loader.load()
            documents.extend(docs)
    
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5100,chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorStore=FAISS.from_documents(documents,embedding=embeddings)
        retriever=vectorStore.as_retriever()
    
        contextualize_system_prompt=(
                "Given a chat history and the latest user question"
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_prompt)
        
        
        system_prompt=(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know.If any question is out of the pdf context," 
                    "dont annswer it and say that it is not given in pdf."
                    "Dont provide any extra information about a topic unless explicitly asked."
                    "Keep the answer restricted to the text given in context"
                    "\n\n"
                    "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
        )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        
        def get_session_history(session:str)->BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id]=ChatMessageHistory()
                return st.session_state.store[session_id]
            
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            # st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            # st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")
