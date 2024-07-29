import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

import streamlit as st

# st.title("Langchain Google Generative AI")

llm = GoogleGenerativeAI(model="gemini-1.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on context provided.

<context>
{context}
</context>

Answer the user in helpful way.

Question: {input}"""
)


def load_file_to_db(uploaded_file):
    loader = PyMuPDFLoader(uploaded_file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_documents = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(
        documents=splitted_documents, embedding=embedding_model
    )
    vector_store.save_local("vectorstore")


def get_vectorstore():
    vectorstore = FAISS.load_local(
        "vectorstore", embeddings=embedding_model, allow_dangerous_deserialization=True
    )
    return vectorstore


def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever_chain = create_retrieval_chain(retriever, doc_chain)

    return retriever_chain


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_query = st.chat_input("Ask a question")

    if user_query:
        st.chat_message("user").write(user_query)
        vectorstore = get_vectorstore()
        qa_chain = create_qa_chain(vectorstore)

        response = qa_chain.invoke({"input": user_query})
        bot_msg = response["answer"]
        st.chat_message("assistant").write(bot_msg)

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_file and st.button("Load file"):
            with open("uploaded_file.pdf", mode="wb") as w:
                w.write(uploaded_file.getvalue())
            load_file_to_db("uploaded_file.pdf")


if __name__ == "__main__":
    main()
