import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

st.set_page_config(page_title="Quiz GPT", page_icon="🤔")

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None
if "difficulty" not in st.session_state:
    st.session_state["difficulty"] = None
if "questions" not in st.session_state:
    st.session_state["questions"] = None


def generate_quiz():
    st.session_state["quiz_id"] = time.time()
    function = {
        "name": "generate_quiz",
        "description": "질문과 답변을 난이도에 맞춰 list 형식으로 10개 의 질문과 4개의 답변을 생성하는 함수",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {"type": "string"},
                                        "correct": {"type": "boolean"},
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },

            },
        },
    }

    llm = ChatOpenAI(
        api_key=st.session_state["openai_api_key"],
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
    ).bind(
        function_call="auto",
        functions=[function]
    )

    prompt = ChatPromptTemplate.from_template(
        """
        당신은 문제 생성을 잘하는 전문 assistant입니다. 
        아래의 요구사항에 따라 문제을 작성해 주세요:
        1. 난이도: {difficulty}
        2. 질문은 한국어로 작성해 주세요.
        3. 다양한 주제를 다룰 수 있도록 문제의 주제를 분배해 주세요. (예:역사, 과학, 일반 상식 등)
        4. 정답이 틀리지 않도록 작성해 주세요.
        """
    )

    chain = prompt | llm

    response = chain.invoke({"difficulty": st.session_state["difficulty"]})
    response = response.additional_kwargs["function_call"]["arguments"]
    import json
    questions = json.loads(response)
    st.session_state["questions"] = questions["questions"]


with st.sidebar:
    st.title("Quiz GPT")
    st.write("난이도와 OpenAI API Key를 입력하면 문제를 생성해줍니다.")
    api_key = st.text_input(
        "OpenAI API Key를 입력해주세요.")
    st.session_state["openai_api_key"] = api_key
    if st.session_state["openai_api_key"]:
        st.session_state["difficulty"] = st.selectbox("난이도 선택", ["쉬움", "어려움"])
        st.button("문제 생성", on_click=generate_quiz)
    st.markdown(
        """
        [Github 링크](https://github.com/kajj8808/gpt-challenge/tree/c96446ed451b2c66ed850097624b958b8ee9a595)
        """
    )

if st.session_state["questions"]:
    with st.form("questions_form"):
        correct_count = 0
        for i, question in enumerate(st.session_state["questions"]):
            answer = st.radio(
                question["question"],
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"quiz_{st.session_state['quiz_id']}_{i}"
            )
            if {"answer": answer, "correct": True} in question["answers"]:
                st.success("정답")
                correct_count += 1
            elif answer is not None:
                st.error("오답")

        is_submit = st.form_submit_button("제출")
        if is_submit:
            if len(st.session_state["questions"]) == correct_count:
                st.balloons()
            correct_count = 0
