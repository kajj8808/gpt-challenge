import streamlit as st
import time

from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS


MESSAGE_SESSION_KEY = "message_history"
OPENAI_SESSION_KEY = "openai_session_key"
MEMORY_SESSION_KEY = "langchain_buffer_memory"

FILES_DIR = "./cache/files"
EMBEDDING_DIR = "./cache/embeddings"


st.set_page_config(
    page_icon="🔥",
    page_title="Streamlit"
)


st.title("Streamlit🔥")

if MEMORY_SESSION_KEY not in st.session_state:
    st.session_state[MEMORY_SESSION_KEY] = ConversationBufferMemory(
        return_messages=True,
    )

if OPENAI_SESSION_KEY not in st.session_state:
    st.session_state[OPENAI_SESSION_KEY] = ""


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, token, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, token, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)
memory = st.session_state[MEMORY_SESSION_KEY]

with st.sidebar:
    st.markdown(""" 
    환영합니다!

    문서 파일에 대해 궁금한 점이 있다면 챗봇에게 물어보세요!

    OpenAI API 키를 입력 후 문서 파일을 업로드하면 대화가 시작됩니다!
    """)

    api_key_input = st.text_input(
        label="Open AI API를 입력해 주세요!",
        type="password",
        disabled=bool(st.session_state[OPENAI_SESSION_KEY]),
        autocomplete="off",
    )

    if api_key_input and not st.session_state[OPENAI_SESSION_KEY]:
        st.session_state[OPENAI_SESSION_KEY] = api_key_input
        st.experimental_rerun()  # 변경이 안되는 문제가 생겨 새로고침!

    if st.session_state[OPENAI_SESSION_KEY]:
        file = st.file_uploader(
            label="`.txt .pdf .docx` 문서 파일을 업로드 해주세요.",
            type=["txt", "pdf", "docx"],
        )


def save_message(message, role):
    st.session_state[MESSAGE_SESSION_KEY].append({
        "message": message,
        "role": role,
    })


def send_mseesage(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(message, role)


def paint_message_history():
    for message in st.session_state[MESSAGE_SESSION_KEY]:
        send_mseesage(message["message"], message["role"], save=False)


@st.cache_data(show_spinner="문서 파일을 분석중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"{FILES_DIR}/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)
    # embedding 된 파일을 cache 할 dir 를 설정해줍니다.
    cache_dir = LocalFileStore(f"{EMBEDDING_DIR}/{file.name}")
    # text를 나눠줍니다.
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings()
    # embeddings 파일로 cache
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "당신은 유용한 도우미입니다. 아래 제공된 문맥만을 사용하여 질문에 답하세요. 답을 모를 경우, 모른다고 말하세요. 답을 지어내지 마세요:\n\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ]
)


def invoke_chain(question):
    chain = {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "history": load_memory
    } | prompt | chat

    result = chain.invoke(question)

    memory.save_context(
        {'input': message},
        {'output': result.content},
    )


if st.session_state[OPENAI_SESSION_KEY]:
    if file:
        retriever = embed_file(file)

        message = st.chat_input(f"{file.name}에 관해 물어보기")
        paint_message_history()
        if message:
            send_mseesage(message, "human")
            with st.chat_message("ai"):
                invoke_chain(message)

    else:
        st.session_state[MESSAGE_SESSION_KEY] = []
