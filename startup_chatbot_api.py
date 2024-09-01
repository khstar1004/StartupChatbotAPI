import os
import json
import sqlite3
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.cache import InMemoryCache
import langchain
from flask import Flask, request, jsonify

import re
import logging

langchain.llm_cache = InMemoryCache()

class EnvironmentSetup:
    @staticmethod
    def load_env(env_path: str):
        load_dotenv(env_path)
        print("Environment variables set successfully.")

class DocumentLoader:
    @staticmethod
    def select_loader(file_path: str):
        if file_path.lower().endswith('.pdf'):
            return PyMuPDFLoader(file_path)
        else:
            return TextLoader(file_path, encoding='utf-8')

    @classmethod
    def load_documents(cls, db_dir: str) -> List[Document]:
        loader = DirectoryLoader(
            db_dir,
            glob="**/*.*",
            loader_cls=cls.select_loader,
            show_progress=True,
            use_multithreading=True
        )
        return loader.load()

    @staticmethod
    def print_document_info(documents: List[Document]):
        print(f"로드된 문서 수: {len(documents)}")
        for i, doc in enumerate(documents, 1):
            print(f"문서 {i}:")
            print(f"  소스: {doc.metadata.get('source', '알 수 없음')}")
            print(f"  내용 일부: {doc.page_content[:100]}...")
            print()

class TextSplitter:
    @staticmethod
    def split_documents(documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100, add_start_index=True
        )
        return text_splitter.split_documents(documents)

class VectorStore:
    @staticmethod
    def create_vectorstore(documents: List[Document]) -> Chroma:
        return Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())

class RAGChain:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    def answer_question(self, question: str) -> str:
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(question)
        context = self._format_docs(retrieved_docs)
        
        prompt = self._create_prompt()
        full_prompt = prompt.format(question=question, context=context)
        answer = self.llm.predict(full_prompt)
        
        return self._format_answer(answer)

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_prompt(self) -> str:
        return """You are an expert in startup advice and entrepreneurship. Use the following context to answer the question. Follow this structure:

        1. Provide a brief, concise summary of the key points (2-3 sentences max).
        2. Then, give a more detailed explanation.

        Answer in Korean.

        Question: {question}
        Context: {context}

        Answer:
        [핵심 요약]
        (여기에 2-3문장으로 핵심 내용을 간단히 요약해주세요.)

        [상세 설명]
        (여기에 더 자세한 설명을 제공해주세요. 단계별 접근법을 사용하고, 중요한 내용은 강조해주세요.)

        친절하고 친근한 톤으로 답변해주세요. 줄바꿈을 적절히 사용하여 가독성을 높여주세요.
        500자 이하로 답해주세요.
        """

    def _format_answer(self, answer: str) -> Dict[str, str]:
        parts = answer.split("[상세 설명]")
        summary = parts[0].replace("[핵심 요약]", "").strip()
        details = parts[1].strip() if len(parts) > 1 else ""
        
        return {
            "summary": summary,
            "details": details
        }

class StartupAdvisorLibrary:
    def __init__(self, db_dir: str, env_path: str):
        EnvironmentSetup.load_env(env_path)
        self.documents = DocumentLoader.load_documents(db_dir)
        splits = TextSplitter.split_documents(self.documents)
        self.vectorstore = VectorStore.create_vectorstore(splits)
        self.rag_chain = RAGChain(self.vectorstore)

    def answer_question(self, question: str) -> Dict[str, str]:
        return self.rag_chain.answer_question(question)

# Flask 애플리케이션 설정
app = Flask(__name__)

# 전역 변수로 StartupAdvisorLibrary 인스턴스 생성
db_dir = "C:/Users/james/Desktop/azicchatbot/documents"  # 실제 문서 디렉토리 경로로 수정
env_path = "C:/Users/james/Desktop/azicchatbot/.env"  # 실제 .env 파일 경로로 수정

# 디렉토리가 존재하는지 확인하고 없으면 생성
import os
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

library = StartupAdvisorLibrary(db_dir, env_path)

@app.route('/api/startup-advice', methods=['POST'])
def get_startup_advice():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "질문이 제공되지 않았습니다."}), 400
    
    answer = library.answer_question(question)
    return jsonify(answer)

if __name__ == "__main__":
    app.run(debug=True)