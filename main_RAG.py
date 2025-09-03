# main_06_fixed.py: 파일 변경 시 RAG 체인이 올바르게 업데이트되도록 수정한 코드
import streamlit as st
# from dotenv import load_dotenv
import os
import tempfile

# LangChain 관련 라이브러리
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 API 키 로드
# load_dotenv()

# --- 1. RAG 시스템을 위한 함수 정의 (기존과 동일) ---
@st.cache_data(show_spinner="PDF 파일을 처리 중입니다...")
def process_pdf(_uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(_uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(documents)
    
    os.remove(tmp_file_path)
    return split_documents

@st.cache_resource(show_spinner="문서 임베딩 및 Retriever를 생성 중입니다...")
def get_retriever(_split_documents):
    embeddings_model = OpenAIEmbeddings()
    db = Chroma.from_documents(_split_documents, embeddings_model)
    return db.as_retriever()

# --- 2. Streamlit UI 설정 및 상태 초기화 ---
st.set_page_config(page_title="Updatable RAG Chatbot", page_icon="🔄")
st.title("🔄 PDF 변경이 가능한 RAG 챗봇")
st.write("PDF를 업로드하고, 다른 파일로 교체하며 채팅을 계속해보세요!")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

uploaded_file = st.sidebar.file_uploader("PDF 파일을 업로드하세요.", type="pdf")

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.current_file_name:
        st.info(f"'{uploaded_file.name}' 파일에 대한 채팅을 시작합니다. 이전 대화 기록이 초기화됩니다.")
        st.session_state.messages = []
        st.session_state.current_file_name = uploaded_file.name

    # --- 3. RAG 파이프라인 실행 및 체인 구성 ---
    
    processed_docs = process_pdf(uploaded_file)
    retriever = get_retriever(processed_docs)
    
    # [핵심 수정] RAG 체인을 세션에 저장하지 않고, 파일이 있을 때마다 새로 구성합니다.
    # 이렇게 하면 retriever가 변경될 때마다 체인도 최신 상태로 업데이트됩니다.
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    prompt_template = ChatPromptTemplate.from_template(
        """
        주어진 컨텍스트만을 사용하여 질문에 답변해주세요. 컨텍스트에 없는 정보는 답변하지 마세요.
        [Context]
        {context}
        [Question]
        {question}
        """
    )
    output_parser = StrOutputParser()
    
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | output_parser
    )
    
    # --- 4. 채팅 인터페이스 구현 ---
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("PDF 내용에 대해 질문해주세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("답변을 생성하는 중입니다..."):
            response = rag_chain.invoke(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
else:
    st.info("사이드바에서 PDF 파일을 업로드하여 챗봇을 시작하세요.")
    st.session_state.messages = []
    st.session_state.current_file_name = None

