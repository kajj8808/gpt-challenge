import openai as client
import streamlit as st
import time
import json
from datetime import datetime
from langchain.tools import DuckDuckGoSearchResults  # link까지 나오는 duckduckgo
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders import AsyncChromiumLoader
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

################################## 데이터 추출 파트 ##################################


def search_duckduckgo(query):

    ddg = DuckDuckGoSearchResults(
        backend="auto", include_links=True, include_text=True, include_images=True, top_k=2)
    return ddg.run(query)


def scrape_website(url):
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs,
        tags_to_extract=[
            "span", "p", "h1", "h2", "h3", "a", "li", "div", "table", "tr", "td", "th"
        ],
    )
    text = "\n\n".join(
        [doc.page_content for doc in docs_transformed]
    )
    return text


def search_wikipedia(query):
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    docs = wiki.run(query)
    return docs


################################## OpenAI function 파트 ##################################
functions_map = {
    "search_wikipedia": search_wikipedia,
    "search_duckduckgo": search_duckduckgo,
    "scrape_website": scrape_website,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "위키피디아에서 주어진 쿼리에 대한 정보를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색하고자 하는 키워드나 문장"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": "DuckDuckGo를 사용하여 웹에서 정보를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색하고자 하는 키워드나 문장"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "웹페이지를 스크래핑 합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "스크래핑할 웹페이지 주소"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

################################## OpenAI Assistant 파트 ##################################


def run_assistant(thread_id, assistant_id):
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def wait_on_run(run_id, thread_id):
    while True:
        run = get_run(run_id, thread_id)
        if run.status == 'queued':
            return run
        elif run.status in ['failed', 'expired', 'cancelled']:
            raise Exception(f"Run failed with status: {run.status}")
        time.sleep(0.5)


def get_response(thread_id):
    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    )
    return messages.data[0].content[0].text.value


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        # json.loads를 사용하는 이유는 응답이 text로 오기에, python에서 사용할 수 있도록 변환 해주는 과정이 필요하기 때문임.
        output = functions_map[function.name](json.loads(function.arguments))
        outputs.append({
            "output": output,
            "tool_call_id": action_id,
        })
    # 결과를 보내기 위해 call id(call_j6Yk2GlAEcoZCfsxTTqn1ARm)들의 결과를 return!
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs
    )


################################## Streamlit 파트 ##################################


if 'thread' not in st.session_state:
    st.session_state.thread = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "assistant" not in st.session_state:
    st.session_state.assistant = None


def paint_state(state):
    if state == 'queued':
        return "**queued** -> requires_action -> completed"
    elif state == 'requires_action':
        return "queued -> **requires_action** -> completed"
    elif state == 'completed':
        return "queued -> requires_action -> **completed**"


def run_assistant_with_message(message):
    # state 표시 queued -> requires_action -> completed
    send_message(st.session_state.thread.id, message)

    run = run_assistant(
        st.session_state.thread.id,
        st.session_state.assistant.id,
    )
    # 상태 표시 이때 현재 상태만 일단.. 간단하게라도 표시
    assistant_state = st.markdown(paint_state(run.status))
    while True:
        run = get_run(run.id, st.session_state.thread.id)
        if run.status == 'requires_action':
            break
        time.sleep(0.5)

    submit_tool_outputs(run.id, st.session_state.thread.id)
    assistant_state.markdown(paint_state(run.status))

    while True:
        run = get_run(run.id, st.session_state.thread.id)
        if run.status == 'completed':
            break
        time.sleep(0.5)
    assistant_state.markdown(paint_state(run.status))

    st.session_state.messages.append(
        {"role": "assistant", "content": get_response(st.session_state.thread.id)})
    with st.chat_message("assistant"):
        st.write(get_response(st.session_state.thread.id))
        st.download_button(
            label="Download Text",
            data=get_response(st.session_state.thread.id),
            file_name="result.txt",
            mime="text/plain",
            # 다운로드 key => 유니크 해야 한다 해서 현재 시간을 키로 사용
            key=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000)}"
        )


def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})


def paint_messages():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])
            if message['role'] == 'assistant':
                st.download_button(
                    label="Download Text",
                    data=message['content'],
                    file_name="result.txt",
                    mime="text/plain",
                    # 다운로드 key => 유니크 해야 한다 해서 현재 시간을 키로 사용
                    key=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000)}"
                )


with st.sidebar:
    st.title("Openai Assistant")
    api_key_input = st.text_input(
        "OpenAI API Key", type="password", value=st.session_state.api_key
    )

    if api_key_input:
        st.session_state.api_key = api_key_input
        client.api_key = api_key_input
        # API 키가 입력되면 thread와 assistant 초기화
        if not st.session_state.thread:
            st.session_state.thread = client.beta.threads.create()
        if not st.session_state.assistant:
            st.session_state.assistant = client.beta.assistants.create(
                name="Research Assistant",
                instructions=""" 
                당신은 검색 AI 에이전트 입니다.
                주어진 주제에 대해 사용할 수 있는 모든 도구를 사용하여 정확한 정보를 수집하세요. 
            
                모든 출처와 URL을 포함해야 합니다.
                """,
                model="gpt-4-turbo-preview",
                tools=functions,
                temperature=0.1,
            )
    st.markdown(
        "[github url](https://github.com/kajj8808/gpt-challenge/tree/ddc8b0957edc12a9441630a010d4ab5d9b768337)")

paint_messages()


if st.session_state.api_key:
    client.api_key = st.session_state.api_key
    message = st.chat_input("Research Assistant!")

    if message:
        with st.chat_message("user"):
            st.write(get_response(st.session_state.thread.id))
        add_message("user", message)
        run_assistant_with_message(message)
