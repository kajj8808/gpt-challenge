
import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def __init__(self, should_stream, should_save):
        self.should_stream = should_stream
        self.should_save = should_save

    def on_llm_start(self, serialized, *args, **kwargs):
        if self.should_stream:
            self.message_box = st.empty()

    def on_llm_new_token(self, token: str, *args, **kwargs):
        if self.should_stream:
            self.message += token
            self.message_box.markdown(self.message)

    def on_llm_end(self, *args, **kwargs):
        if self.should_save:
            save_message("ai", self.message)


if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

messages = st.session_state["messages"]
openai_api_key = st.session_state["openai_api_key"]

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", value="", type="password")

    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key

st.title("Site GPT")


answers_prompt = ChatPromptTemplate.from_template(
    """
   Using ONLY the provided context, answer the user's question. If the context does not contain the necessary information, respond with "I don't know" and do not fabricate any information.

    After providing your answer, assign a score between 0 and 5 based on how well your response addresses the user's question:
    - 5: Fully answers the question.
    - 4: Mostly answers the question.
    - 3: Partially answers the question.
    - 2: Minimally answers the question.
    - 1: Barely answers the question.
    - 0: Does not answer the question or is unrelated.

    Always include the score, even if it's 0.

    Context: {context}  
    Question: {question}

    Answer:

    """)


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def parse_page(soup):
    header = soup.find("header")
    if header:
        header.decompose()
    return soup.get_text()


def get_llm(should_stream=False, should_save=False):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0.1,
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[ChatCallbackHandler(
            should_stream=should_stream,
            should_save=should_save,
        )],
    )

    return llm


@st.cache_data(show_spinner="Loading site content...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[
            "https://developers.cloudflare.com/ai-gateway/",
            "https://developers.cloudflare.com/vectorize/",
            "https://developers.cloudflare.com/workers-ai/"
        ],
        parsing_function=parse_page,
    )

    docs = loader.load_and_split(splitter)
    vector_store = FAISS.from_documents(
        docs, OpenAIEmbeddings(openai_api_key=openai_api_key),
    )
    return vector_store.as_retriever()


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    llm = get_llm()
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answer": [
            {
                "answer": answers_chain.invoke({
                    "context": doc.page_content,
                    "question": question
                }).content,
                "source": doc.metadata["source"],
                "lastmod": doc.metadata["lastmod"],
            } for doc in docs
        ]
    }


def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answer"]
    llm = get_llm(
        should_stream=True,
        should_save=True,
    )

    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"Answer: {answer['answer']}\nSource: {answer['source']}\nLastmod: {answer['lastmod']}\n" for answer in answers
    )

    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def paint_message_history():
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def save_message(role, content):
    messages.append({"role": role, "content": content})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(role, message)


if openai_api_key:
    paint_message_history()

    retriever = load_website("https://developers.cloudflare.com/sitemap-0.xml")

    query = st.chat_input("Cloudflare공식 문서에 대해서 궁금한 점을 물어봐 보세요 !")

    if query:
        # st.write(retriever.invoke(query))
        send_message(query, "user")
        with st.chat_message("ai"):
            chain = {
                "docs": retriever,
                "question": RunnablePassthrough(),
            } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
            result = chain.invoke(query)
