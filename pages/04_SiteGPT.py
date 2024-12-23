import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer

st.title("QuizGPT")

html2text_transformer = Html2TextTransformer()


with st.sidebar:
    url = st.text_input("url 을 입력해 주세요.!", placeholder="https://example.com")

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(docs)
