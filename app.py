import streamlit as st
import time

MESSAGE_SESSION_KEY = "message_history"
OPENAI_SESSION_KEY = "openai_session_key"

FILES_DIR = "./cache/files"

st.set_page_config(
    page_icon="🔥",
    page_title="Streamlit"
)


st.title("Streamlit🔥")


# 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.

if OPENAI_SESSION_KEY not in st.session_state:
    st.session_state[OPENAI_SESSION_KEY] = ""


with st.sidebar:
    st.markdown(""" 
    환영합니다!

    문서 파일에 대해 궁금한 점이 있다면 챗봇에게 물어보세요!

    OpenAI API 키를 입력 후 문서 파일을 업로드하면 대화가 시작됩니다!
    """)

    # API 키 입력
    api_key_input = st.text_input(
        label="Open AI API를 입력해 주세요!",
        type="password",
        disabled=bool(st.session_state[OPENAI_SESSION_KEY]),
        autocomplete="off",
    )

    if api_key_input and not st.session_state[OPENAI_SESSION_KEY]:
        st.session_state[OPENAI_SESSION_KEY] = api_key_input
        st.experimental_rerun()  # 변경 사항을 반영하기 위해 새로고침

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


if st.session_state[OPENAI_SESSION_KEY]:
    if file:
        file_content = file.read()
        file_path = f"{FILES_DIR}/{file.name}"

        with open(file_path, "wb") as f:
            f.write(file_content)

        message = st.chat_input(f"{file.name}에 관해 물어보기")
        paint_message_history()
        if message:
            send_mseesage(message, "human")
            time.sleep(2)
            send_mseesage("hahaha", "ai")
    else:
        st.session_state[MESSAGE_SESSION_KEY] = []
