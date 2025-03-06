from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter  # Import the correct text splitter

import pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Class for Resume Analysis ChatBot
class ChatBot():
    def __init__(self):
        # Document loading and splitting
# Document loading and splitting
        self.loader = TextLoader(r'C:/Users/parth/Desktop/Projects/Lanchain_Basic_Bot/Basic_Rag_Chatbot/resume.txt', encoding='utf-8')
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


        # Initialize Pinecone vector store
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
        self.index_name = "resume-analysis-demo"

        # Check if the index exists, else create it
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.index_name, metric="cosine", dimension=768)
        self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)

        # Initialize LLM (using a model from Hugging Face Hub)
        self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id,
            model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Define prompt template for resume analysis
        self.template = """
        You are a resume analyzer bot. Your task is to analyze resumes and provide concise feedback.
        Consider skills, experience, and job relevance when assessing the resume. If the resume is unclear in certain areas, mention that.
        If you don't know something, just say you don't know.

        Context: {context}
        Question: {question}
        Answer: 
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

        # Set up the Retrieval-Augmented Generation (RAG) chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.docsearch.as_retriever(),
            chain_type="map_reduce",  # You can experiment with different chain types like 'stuff', 'map_reduce', etc.
        )

    def generate_response(self, question: str):
        # Use the RAG chain to get the answer based on the resume context and the question
        response = self.rag_chain.run(question)
        return response
