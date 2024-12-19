import streamlit as st
import time

MESSAGE_SESSION_KEY = "message_history"
FILES_DIR = "./cache/files"

st.set_page_config(
    page_icon="🔥",
    page_title="Streamlit"
)

st.title("Streamlit🔥")

with st.sidebar:
    st.markdown(""" 
    환영합니다!

    문서 파일에 대해 궁금한 점이 있다면 챗봇에게 물어보세요!

    문서 파일을 업로드하면 대화가 시작됩니다!
    """)
    file = st.file_uploader(
        label="`.txt .pdf .docx` 문서 파일을 업로드 해주세요.",
        type=[".txt", ".pdf", ".docx"],
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
# 파일 업로드 및 채팅 기록을 구현합니다.
