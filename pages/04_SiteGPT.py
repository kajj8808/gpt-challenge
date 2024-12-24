import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
)


llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template("""
        Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                    
        Then, give a score to the answer between 0 and 5.
        If the answer answers the user question the score should be high, else it should be low.
        Make sure to always include the answer's score even if it's 0.
        Context: {context}
                                                    
        Examples:
                                                    
        Question: How far away is the moon?
        Answer: The moon is 384,400 km away.
        Score: 5
                                                    
        Question: How far away is the sun?
        Answer: I don't know
        Score: 0
                                                    
        Your turn!                                     
        
        Question: {question}
    """)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []

    # for doc in docs:
    # result = answers_chain.invoke({
    # "question": question,
    # "context": doc.page_content
    # })
    # answers.append(result.content)
    return {
        "question": question,
        "answers":  [
            {
                "answer": answers_chain.invoke({
                    "question": question,
                    "context": doc.page_content
                }).content,
                "source": doc.metadata["source"],
            } for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
        먼저 생성된 answers들만을 사용하여 사용자의 question에 답변하세요.
        더 높은 점수를 가진 답변들을 사용하세요.(더 유용합니다.)

        최신의 자료를 우선시하고, 출처도 남겨주세요. 

        Answers : {answers}
        """),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"Answer:{answer['answer']}\nSource:{answer['source']}\n"
        for answer in answers)

    return choose_chain.invoke(
        {
            "question": question,
            "answers": answers,
        },
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")

    if header:
        header.decompose()
    if footer:
        footer.decompose()

    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Blog Submit", " ")
    )


@ st.cache_data(show_spinner="loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page
    )
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.title("Site GPT")


with st.sidebar:
    url = st.text_input("url 을 입력해 주세요.!", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Sitemap URL을 입력해주세요.(.xml)")
    else:
        retriever = load_website(url)
        query = st.text_input("사이트에 대해서 궁금하신게 있다면 물어보세요!")
        # docs = retriever.invoke("Robin 과 관련된 제품이 있나요?")
        if query:
            chain = {
                "docs": retriever,
                "question": RunnablePassthrough(),
            } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

            result = chain.invoke(query)
            st.write(result.content)
            # chain.invoke("이어폰에 관련된 제품이 어떤게 있나요?")

# https://moondroplab.com/sitemap.xml
