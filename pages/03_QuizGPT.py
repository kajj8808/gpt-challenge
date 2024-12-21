import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        import json
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🤔"
)

st.title("Quiz GPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            당신은 교사역활을 하는 assistant 입니다.
            아래의 내용에 따라 유저의 지식을 테스트하기 위해 10개의 질문을 만드세요.

            각 문제에는 4개의 답이 있어야 하며, 그 중 3개는 틀린 답이고, 1개는 정답이어야 합니다.

            오직 (o)를 사용하여 정답을 표시합니다.

            Question examples:

            Question: 바다의 색깔은 무엇인가요?.
            Answer: 빨강|노랑|초록|파랑(o)

            Question: 조지아의 수도는 어디인가요?
            Answer: Baku|Tbilsi(o)|Manila|Beirut

            Question: 영화 Avatar는 언제 개봉했나요?
            Answer: 2007|2001|2009(o)|1998

            Question: Julius Caesar는 누구인가요?
            Answer: 로마 황제(o)|화가|배우|모델


            Your turn!

            Context: {context}
             """)
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        당신은 강력한 formatting 알고리즘입니다.

        시험 문제를 JSON 형식으로 포맷합니다. (o)가 표시된 답이 정답입니다.

        Example Input:

        Question: 바다의 색깔은 무엇인가요?
        Answers: 빨강|노랑|초록|파랑(o)

        Question: 조지아의 수도는 어디인가요?
        Answer: Baku|Tbilsi(o)|Manila|Beirut


        Question: 영화 Avatar는 언제 개봉했나요?
        Answers: 2007|2001|2009(o)|1998

        Question: Julius Caesar는 누구인가요?
        Answer: 로마 황제(o)|화가|배우|모델

        Example Output:

        ```json
        {{
            "questions": [
                {{
                    "question": "바다의 색깔은 무엇인가요?",
                    "answers": [
                        {{ "answer": "빨강", "correct": false }},
                        {{ "answer": "노랑", "correct": false }},
                        {{ "answer": "초록", "correct": false }},
                        {{ "answer": "파랑", "correct": true }}
                    ]
                }},
                {{
                    "question": "조지아의 수도는 어디인가요?",
                    "answers": [
                        {{ "answer": "Baku", "correct": false }},
                        {{ "answer": "Tbilsi", "correct": true }},
                        {{ "answer": "Manila", "correct": false }},
                        {{ "answer": "Beirut", "correct": false }}
                    ]
                }},
                {{
                    "question": "영화 Avatar는 언제 개봉했나요?",
                    "answers": [
                        {{ "answer": "2007", "correct": false }},
                        {{ "answer": "2001", "correct": false }},
                        {{ "answer": "2009", "correct": true }},
                        {{ "answer": "1998", "correct": false }}
                    ]
                }},
                {{
                    "question": "율리우스 카이사르는 누구인가요?",
                    "answers": [
                        {{ "answer": "로마 황제", "correct": true }},
                        {{ "answer": "화가", "correct": false }},
                        {{ "answer": "배우", "correct": false }},
                        {{ "answer": "모델", "correct": false }}
                    ]
                }}
            ]
        }}
        ```
        Your turn!

        Questions: {context}
""",
    )
])

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipidia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=1)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipeida...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown("""
    Quiz GPT에 오신걸 환영합니다.

    위키백과 기사나 파일에서 퀴즈를 만들어 지식을 테스트하고, 공부하는 데 도움을 드리겠습니다.
                
    사이드바에 파일을 업로드하거나, 위키백과에서 검색을 하는 것으로 시작하세요.!
                
    """)
else:

    response = run_quiz_chain(docs, topic if topic else file.name)

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an options",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("정답")
            elif value is not None:
                st.error("오답")
        button = st.form_submit_button()
