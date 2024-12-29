import openai as client
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.tools import DuckDuckGoSearchResults  # link까지 나오는 duckduckgo
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders import AsyncChromiumLoader
################################## 데이터 추출 파트 ##################################


@st.cache_data(show_spinner="웹 찾아보는중...")
def search_duckduckgo(query):
    ddg = DuckDuckGoSearchResults()
    return ddg.run(query)


@st.cache_data(show_spinner="웹 페이지 찾아보는중...")
def webpage_scraper(url):
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


@st.cache_data(show_spinner="위키피디아 검색중...")
def search_wikipedia(query):
    retriever = WikipediaRetriever(
        top_k_results=1,
    )
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])


################################## OpenAI function 파트 ##################################
functions_map = {
    "search_wikipedia": search_wikipedia,
    "search_duckduckgo": search_duckduckgo,
    "webpage_scraper": webpage_scraper,
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
            "name": "search_web",
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
            "name": "webpage_scraper",
            "description": "웹페이지를 스크래핑하여 결과를 반환합니다.",
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
################################## OpenAI Agent 파트 ##################################


@st.cache_data(show_spinner="AI 에이전트 생성중...")
def get_assistant():
    assistant = client.beta.assistants.create(
        name="Search Assistant",
        instructions="당신은 사용자의 질문에 대해 적절한 도구를 선택하여 검색을 수행합니다.",
        model="gpt-4o-mini",
        tools=functions,
    )
    return assistant


@st.cache_data(show_spinner="스레드 생성중...")
def create_thread():
    thread = client.beta.threads.create()
    return thread


def add_message_to_thread(thread_id, message):
    return client.beta.threads.messages.create(thread_id, role="user", content=message)


def run_assistant(thread_id, assistant_id, message):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=message,
    )
    return run


def wait_on_run(run_id, thread_id):
    import time
    while True:
        run_status = get_run_status(run_id=run_id, thread_id=thread_id)
        if run_status.status == "completed":
            break
        time.sleep(0.5)
    return run_status


def get_run_status(run_id, thread_id):
    return client.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)
