import os
import logging
from typing import List, Dict, Any

import bs4
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self):
        self.load_environment()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.vectorstore = None
        self.initialize_vectorstore()
        
    def load_environment(self):
        """Load environment variables from .env file"""
        try:
            load_dotenv(dotenv_path="os.env")
            self.openai_api_key = os.getenv("OPENAI_API_KEY")           
            self.user_agent = os.getenv("USER_AGENT")
            os.environ['USER_AGENT'] = self.user_agent
            logger.info("Environment variables loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading environment variables: {e}")
            raise
            
    def initialize_vectorstore(self):
        """Initialize the vector store with documents"""
        try:
            # Define default URLs for knowledge base
            default_urls = [
                "https://www.linkedin.com/pulse/insights-post-pandemic-economy-our-2024-global-market-rob-sharps-jcnmc/"
            ]
            
            # Load documents from web
            loader = WebBaseLoader(
                web_paths=default_urls,
                header_template={"User-Agent": self.user_agent}
            )
            logger.info(f"Loading documents from {len(default_urls)} default URLs")
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            logger.info(f"Split documents into {len(splits)} chunks")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            )
            
            logger.info(f"Vector store initialized with {len(splits)} document chunks")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, urls: List[str]):
        """Add new documents to the knowledge base"""
        try:
            if not urls:
                return {"status": "error", "message": "No URLs provided"}
                
            logger.info(f"Adding documents from {len(urls)} URLs")
            
            # Ensure vectorstore is initialized
            if self.vectorstore is None:
                self.initialize_vectorstore()
                return {"status": "success", "message": "Vector store initialized with default documents"}
                
            # Load and process new URLs
            loader = WebBaseLoader(
                web_paths=urls,
                header_template={"User-Agent": self.user_agent}
            )
            
            try:
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} new documents")
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                logger.info(f"Split new documents into {len(splits)} chunks")
                
                # Add to existing vectorstore
                self.vectorstore.add_documents(splits)
                
                return {"status": "success", "message": f"Added {len(splits)} document chunks to knowledge base"}
            except Exception as loading_error:
                logger.error(f"Error loading documents: {loading_error}")
                return {"status": "error", "message": f"Error loading documents: {str(loading_error)}"}
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"status": "error", "message": str(e)}
            
    def get_response(self, user_query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Get response for user query using RAG pipeline"""
        try:
            if not user_query.strip():
                return {"answer": "Please provide a question or message.", "sources": []}
                
            logger.info(f"Processing query: {user_query[:50]}{'...' if len(user_query) > 50 else ''}")
                
            # Convert chat history format if provided
            formatted_history = []
            if chat_history:
                for msg in chat_history:
                    if msg.get("role") == "user":
                        formatted_history.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        formatted_history.append(AIMessage(content=msg.get("content", "")))
            
            # Ensure vectorstore is initialized
            if self.vectorstore is None:
                self.initialize_vectorstore()
            
            # Set up retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Create the LLM
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                temperature=0.2, 
                api_key=self.openai_api_key
            )
            
            # Custom RAG prompt template
            template = """You are a helpful and knowledgeable assistant providing accurate information based on the context provided.
            
            Use the following pieces of context to answer the user's question:
            
            {context}
            
            If you don't know the answer based on the context, say that you don't know rather than making up information.
            Always be honest about your knowledge limitations.
            Provide concise, easy-to-understand answers that directly address the user's question.
            When appropriate, structure your response with bullet points or numbered lists for clarity.
            
            Question: {question}
            
            Helpful answer:"""
            
            custom_rag_prompt = PromptTemplate.from_template(template)
            
            # Create a conversational chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": custom_rag_prompt}
            )
            
            # Get response
            result = qa_chain({"question": user_query})
            
            # Extract source information
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:150] + "...",
                        "source": doc.metadata.get("source", "Unknown")
                    }
                    sources.append(source_info)
            
            logger.info(f"Generated response with {len(sources)} source references")
            
            return {
                "answer": result["answer"],
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"answer": f"I encountered an error while processing your request. Please try again later.", "sources": []}

# Initialize the chatbot service for direct imports
chatbot_service = ChatbotService()

def response(user_query):
    """Legacy function for compatibility with existing code"""
    result = chatbot_service.get_response(user_query)
    return result["answer"]
