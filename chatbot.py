from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
import pinecone
from dotenv import load_dotenv
import os
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

# Class for Resume Analysis ChatBot
class ChatBot():

    def __init__(self):
        # Document loading and splitting
        self.loader = TextLoader('./resume.txt')  # Change file name as necessary
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings()

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
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()} 
            | self.prompt 
            | self.llm
            | StrOutputParser()
        )

    def generate_response(self, question: str):
        # Use the RAG chain to get the answer based on the resume context and the question
        response = self.rag_chain.invoke({"question": question})
        return response
