import streamlit as st
from langchain.document_loaders import SitemapLoader


@st.cache_data(show_spinner="loading website...")
def load_website(url):
    loader = SitemapLoader(url)
    docs = loader.load()
    return docs


st.title("QuizGPT")


with st.sidebar:
    url = st.text_input("url 을 입력해 주세요.!", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Sitemap URL을 입력해주세요.(.xml)")
    else:
        docs = load_website(url)
        st.write(docs)
