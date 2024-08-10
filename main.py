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

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


import streamlit as st


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on context provided.

<context>
{context}
</context>

Explain the user from context in detail.
Use bookish language with bullet points.
Do not add unnecessary title or headings and use simple but professional language.
When it ask about working of topic or its application then also add explanation of that topic and then anwer the question.
Also try to provide diagram or illustration if possible.

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
    try:
        vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        return vectorstore
    except Exception:
        return None


def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever_chain = create_retrieval_chain(retriever, doc_chain)

    return retriever_chain


def get_query_response(query, vectorstore):
    qa_chain = create_qa_chain(vectorstore)
    response = qa_chain.invoke({"input": query})
    return response["answer"]


def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.title("Chat with PDF using Gemini💁")

    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    for msg in st.session_state["message_history"]:
        st.chat_message(msg[0]).write(msg[1])

    user_query = st.chat_input("Ask a question")

    if user_query:
        st.chat_message("user").write(user_query, unsafe_allow_html=True)
        vectorstore = get_vectorstore()

        if vectorstore:
            response = get_query_response(user_query, vectorstore)
            bot_msg = response
        else:
            bot_msg = (
                "No context available. Please upload a PDF file to provide context."
            )

        st.chat_message("assistant").write(bot_msg)

        st.session_state["message_history"].append(("user", user_query))
        st.session_state["message_history"].append(("assistant", bot_msg))

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload a PDF file", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Load file"):
            for index, uploaded_file in enumerate(uploaded_files):
                with open(f"uploaded_file_{index}.pdf", mode="wb") as w:
                    w.write(uploaded_file.getvalue())
                load_file_to_db(f"uploaded_file_{index}.pdf")


if __name__ == "__main__":
    main()
