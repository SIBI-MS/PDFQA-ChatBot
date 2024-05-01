import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage #schemas
from langchain_community.document_loaders import WebBaseLoader  #load content from url
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter #used make chunks using the contents from url
from langchain_community.vectorstores import Chroma #to store vectors created my using the above chunks
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

# #Storing apikey and model name
# os.environ['HUGGINGFACE_API_KEY']=os.getenv("HUGGINGFACE_API_KEY")
# model_id="meta-llama/Llama-2-7b-chat-hf"


def get_vectorstore_from_url(pdf_docs):
    # get the text in document form
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    # return text
    
    # split the document into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    # chunks = text_splitter.split_text(text)
    document_chunks = text_splitter.split_text(text)
    print(type(document_chunks))

    
    # create a vectorstore from the chunks
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma.from_texts(document_chunks,embeddings)

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = Ollama(
        model="phi3",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

#For create a chain
def get_conversational_rag_chain(retriever_chain): 
    # llm = HuggingFaceHub(
    #     huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
    #     repo_id=model_id,
    #     model_kwargs={"temperature": 0.3, "max_new_tokens": 500}
    # )
    llm = Ollama(
        model="phi3",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain,stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="ChatWeb", page_icon="🤖")
st.title("ChatWeb")

# sidebar
with st.sidebar:
    st.header("Settings")
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

    if pdf_docs is None:
        st.info("Upload Your PDFs Here")

    # elif st.button("Process"):
    #     with st.spinner("Processing"):
    # session state
    if st.button("Process"):
        with st.spinner("Processing"):
            if "vector_store" not in st.session_state: 
                st.session_state.vector_store = get_vectorstore_from_url(pdf_docs) 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ] 


# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    


# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)