import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# --- IMPORTS (Latest Standards) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Caching embeddings to avoid reloading on every run -- important for performance
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")


load_dotenv()

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI-Powered Document Intelligence (RAG)",
    page_icon="📄",
    layout="centered"
)

st.title("AI-Powered Document Intelligence (RAG)")
st.caption("Upload a document and ask questions from it")

# ---------- SIDEBAR ----------
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"]
)

# Clear conversation button
if st.sidebar.button("Clear Conversation History"):
    st.session_state.conversation_history = []

# ---------- LLM ----------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
    #model_name="llama-3.3-70b-versatile"   #vers complex for pdf but slow compared to llama-3.1
)

# ---------- SESSION STATE ----------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Track last processed file to avoid reprocessing on reruns
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None


# ---------- PROCESS DOCUMENT ----------
if uploaded_file:
    # Build a simple file id so the same upload is not reprocessed on reruns
    file_size = getattr(uploaded_file, "size", None)
    file_id = f"{uploaded_file.name}-{file_size}"

    if st.session_state.last_file_id != file_id:
        with st.spinner("Processing document..."):
            # Must save Temp file for loaders
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 1. Load Document
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)

            documents = loader.load()

            # 2. Split Text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(documents)
            st.info(f"Split into {len(chunks)} chunks") #display number of chunks

            # 3. Create Embeddings
            
            embeddings = load_embeddings()


            #embeddings = HuggingFaceEmbeddings(
            #    model_name="sentence-transformers/all-MiniLM-L6-v2"
            #)
            


            # 4. Create Vector Store
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # 5. Create QA Chain
            prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say "I don't know". Do NOT try to make up an answer or use information not in the context.

{context}

Question: {question}
Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",                        # ناخد كل ال chunks في مرة واحدة (ممكن نغيرها لو عايزين ناخد chunk واحد بس في كل مرة)
                                                        # ممكن نستخدم refine هيجيب اجابات دقيقه more structured.
                                                        # map_reduce ده ملفات ضخمه 
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),       #  بدون استخدامه هيطلع اجابه عشوائية بدون ما يستخدم الملف و استخدمت K=3 علشان يرجع افضل3 اجزاء 
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            st.session_state.qa_chain = qa_chain
            st.session_state.last_file_id = file_id

            st.success("Document processed successfully!")
    # else:
    #     st.info("Document already processed. You can continue chatting.")


# ---------- CHAT ----------
if st.session_state.qa_chain:
    # Display conversation history
    if st.session_state.conversation_history:
        for idx, item in enumerate(st.session_state.conversation_history, 1):
            st.markdown(f"**Q{idx}:** {item['question']}")
            st.markdown(f"**A{idx}:** {item['answer']}")

            if item.get("sources") and "don't know" not in item["answer"].lower():
                with st.expander("View Sources"):
                    for i, doc in enumerate(item["sources"]):
                        page_number = doc["page"]
                        content = doc["content"]
                        st.markdown(f"**Source {i+1} (Page {page_number})**")
                        st.markdown(content)
                        st.divider()

    
    # Input for new question
    user_question = st.chat_input("Ask a question from the document...")

    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({"query": user_question})
            answer = response["result"]
            sources = response["source_documents"]

        # Store in conversation history
        safe_sources = []
        for doc in sources:
            safe_sources.append({
                "page": doc.metadata.get("page", "Unknown"),
                "content": doc.page_content[:500]  # اختصر لو طويل
    })

        st.session_state.conversation_history.append({
            "question": user_question,
            "answer": answer,
            "sources": safe_sources
        })

        # Display the new response
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            st.markdown(answer)


            if sources and "don't know" not in answer.lower() :
                with st.expander("View Sources"):
                    for i, doc in enumerate(safe_sources):
                        page_number = doc["page"]
                        content = doc["content"]
                        st.markdown(f"**Source {i+1} (Page {page_number})**")
                        st.markdown(content)
                        st.divider()
else:
    st.info("Please upload a document to start.")

