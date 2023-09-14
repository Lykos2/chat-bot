
import os 
import io 
import torch 
import streamlit as st
from PyPDF2 import PdfReader
from helper import get_file_list
from helper import read_file_to_list
from helper import process_pdf
from helper import load_model
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain import FAISS


device="cuda" if torch.cuda.is_available() else "cpu"

device_type="cuda" if torch.cuda.is_available() else "cpu"
cwd = os.getcwd()



Model_file_path = os.path.join(cwd, "models.txt")
Embedding_file_path = os.path.join(cwd, "Emb.txt")
os.makedirs(os.path.dirname(Model_file_path), exist_ok=True)
os.makedirs(os.path.dirname(Embedding_file_path), exist_ok=True)






with st.sidebar:
    st.title('🦙💬 PDF-Chatbot')
    uploaded_files = st.file_uploader("Choose a PDF File", accept_multiple_files=True)
    MODEL = st.selectbox(
        'Available Hugging Face models ',
        ( read_file_to_list(Model_file_path)))
    if MODEL =="other":
        MODEL= st.text_input("Enter the link of the Hugging Face model:")

            
    EMB = st.selectbox(
        'Available Hugging Face Embeddings ',
        ( read_file_to_list(Embedding_file_path)))
    if EMB =="other":
        EMB= st.text_input('# Link of huggingface Embeddings',)




    files=[]
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in files:
            bytes_data = uploaded_file.read()
            pdf_stream = io.BytesIO(bytes_data)
            pdf_reader = PdfReader(pdf_stream)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text()
            if 'text' not in st.session_state:
                st.session_state['text']=text 
            else:
                st.session_state['text']+=text
            files.append(uploaded_file.name)
        else:
            pass
    if EMB=="":
        st.write('please select the Emb')
    elif MODEL=="":
        st.write('please select the model')
    else:
        
        st.write('Model selected:', MODEL)
        st.write('Embedding selected:', EMB)

    result=st.button("Done")

if result:
    if MODEL!="":
   
        MODEL_BASENAME="llama-2-7b-chat.ggmlv3.q4_0.bin"
        llm = load_model(device_type, model_id=MODEL, model_basename=MODEL_BASENAME)
        st.session_state['llm']=llm
   
  
    embedding=HuggingFaceEmbeddings(
                model_name=EMB,
                model_kwargs={'device':device},
                encode_kwargs={'normalize_embeddings':False}
            )
    if 'embedding' not in st.session_state:
        st.session_state['emb']=embedding

        

    text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

    chunks=text_splitter.split_text(st.session_state['text'])
    
    faiss = FAISS.from_texts(chunks, st.session_state['emb'])
    if 'faiss' not in st.session_state:
        st.session_state['faiss']=faiss


    
    retriever = st.session_state['faiss']
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

  
    if 'QA'  not in st.session_state:
        qa = RetrievalQA.from_chain_type(
        llm=st.session_state['llm'],
        chain_type="stuff",
        retriever = retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
        st.session_state["QA"] = qa




prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
    response = st.session_state["QA"](prompt)
    answer, docs = response["result"], response["source_documents"]
    st.write(answer)
