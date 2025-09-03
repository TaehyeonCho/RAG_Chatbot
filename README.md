# RAG_Chatbot
PDF RAG 챗봇 with Streamlit
이 프로젝트는 Streamlit을 사용하여 만든 RAG(Retrieval-Augmented Generation) 기반의 PDF 챗봇 웹 애플리케이션입니다. 
PDF 파일을 업로드하고, 문서 내용에 대해 AI와 실시간으로 대화할 수 있습니다.

1. 주요 기능
PDF 파일 업로드: 웹 인터페이스를 통해 PDF 문서를 쉽게 업로드할 수 있습니다.

실시간 RAG 파이프라인: 업로드된 문서는 실시간으로 텍스트로 분할되고, OpenAI 임베딩 모델을 통해 벡터로 변환되어 메모리 기반의 ChromaDB에 저장됩니다.

대화형 챗봇: LangChain Expression Language (LCEL)로 구성된 체인을 통해 문서 내용에 기반한 질문에 답변합니다.

동적 DB 업데이트: 다른 PDF 파일로 교체하면, 이전 대화 기록과 데이터베이스가 자동으로 초기화되고 새로운 문서에 대한 챗봇이 활성화됩니다.

2. 동작 원리
파일 업로드: 사용자가 Streamlit 웹 UI를 통해 PDF 파일을 업로드합니다.

문서 처리: PyPDFLoader로 문서를 로드하고 RecursiveCharacterTextSplitter를 사용해 적절한 크기의 텍스트 조각으로 분할합니다.

임베딩 및 저장: OpenAIEmbeddings를 사용해 각 텍스트 조각을 벡터로 변환하고, Chroma 벡터 스토어에 메모리 내에 저장합니다.

검색 및 답변 생성: 사용자가 질문을 하면, 질문과 의미적으로 유사한 텍스트 조각을 DB에서 검색(Retrieve)하고, 검색된 내용과 원본 질문을 LLM(GPT-3.5-Turbo)에 함께 전달하여 최종 답변을 생성(Generate)합니다.

3. 시작하기
사전 준비
Python 3.8 이상

OpenAI API 키

4. 설치 방법
프로젝트 저장소 복제 (Clone)

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

환경 변수 설정
프로젝트 루트 디렉토리에 .env 파일을 생성하고, 아래와 같이 OpenAI API 키를 입력하세요.

OPENAI_API_KEY="sk-..."

필요한 라이브러리 설치
아래 명령어를 실행하여 requirements.txt 파일에 명시된 모든 라이브러리를 설치합니다.

pip install -r requirements.txt

실행 방법
터미널에서 아래 명령어를 실행하여 Streamlit 웹 서버를 시작합니다.

streamlit run main_06.py

명령어 실행 후, 웹 브라우저에 나타나는 로컬 주소(예: http://localhost:8501)로 접속하세요.
