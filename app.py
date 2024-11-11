import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import InferenceApi
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
import google.generativeai as genai
import PIL.Image
import os
import getpass
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GOOGLE_API_KEY = "AIzaSyDTXmOEOdQpjZ2-ld03cDUSBTMbZ9jR4uE"

# HuggingFace API Key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore_from_pdf(text_chunks):
    # create a vectorstore from the chunks
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma.from_texts(text_chunks,embeddings)

    return vector_store

def get_context_retriever_chain(vector_store):
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1,
    max_tokens=512)
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Based on the above conversation, generate a search query to look up relevant information in the provided context and also you should behave like a cyberark expert. Do not guess or use any outside knowledge except information about cyberark.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1,
    max_tokens=512)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "you should behave like a cyberark expert and Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# Streamlit app configuration
st.set_page_config(page_title="PDFQA", page_icon="📕")
st.title("PDFQA📚")

with st.sidebar:
    st.header("Settings")
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            if "vector_store" not in st.session_state:
                st.session_state.vector_store = get_vectorstore_from_pdf(text_chunks)

            

if pdf_docs is None:
    st.info("Please Upload PDFs")

        
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello...")
        ]

    # Handle user input
    user_query = st.chat_input("Ask questions about your pdf..")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    
    def extract_relevant_content(message):
        if isinstance(message, AIMessage):
            # Extract the first MD content from the AI message
            content = message.content
            start_idx = content.find("MD:")
            if start_idx != -1:
                return content[start_idx + len("MD:"):].strip()
        return message.content

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(extract_relevant_content(message))
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)




# llm = HuggingFaceHub(
    #     repo_id="meta-llama/Llama-2-7b", 
    #     model_kwargs={"temperature": 0.1, "max_length": 512}, 
    #     huggingfacehub_api_token=HUGGINGFACE_API_KEY
    # )
    # genai.configure(api_key="AIzaSyDuG5D2Sc6WCRtZaiofP9LM_BxxhDD8Q7A")
    # llm = genai.GenerativeModel(model_name="gemini-1.5-flash")