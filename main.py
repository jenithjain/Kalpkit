#!/usr/bin/env python3
"""
AmbedkarGPT - Q&A System based on Dr. B.R. Ambedkar's Speech
A RAG (Retrieval-Augmented Generation) system using LangChain, ChromaDB, and Ollama

"""

import os
import sys
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document


class AmbedkarGPT:
    """
    A Q&A system that answers questions based on Dr. B.R. Ambedkar's speech content.
    Uses RAG pipeline with ChromaDB for vector storage and Ollama for LLM inference.
    """
    
    def __init__(self, speech_file: str = "speech.txt", persist_directory: str = "./chroma_db"):
        """
        Initialize the AmbedkarGPT system.
        
        Args:
            speech_file (str): Path to the speech text file
            persist_directory (str): Directory to persist ChromaDB data
        """
        self.speech_file = speech_file
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize embeddings model
        print("📚 Loading HuggingFace embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Ollama LLM
        print("🤖 Connecting to Ollama Mistral 7B Quantized...")
        self.llm = Ollama(
            model="mistral:7b-instruct-q4_K_M",
            temperature=0.1  # Low temperature for more focused answers
        )
        
    def load_and_process_text(self) -> List[Document]:
        """
        Load the speech text file and split it into manageable chunks.
        
        Returns:
            List[Document]: List of document chunks
        """
        print(f"📖 Loading text from {self.speech_file}...")
        
        # Check if file exists
        if not os.path.exists(self.speech_file):
            raise FileNotFoundError(f"Speech file '{self.speech_file}' not found!")
        
        # Load the document
        loader = TextLoader(self.speech_file, encoding='utf-8')
        documents = loader.load()
        
        # Split text into chunks
        print("✂️  Splitting text into chunks...")
        text_splitter = CharacterTextSplitter(
            chunk_size=200,  # Smaller chunks for better retrieval
            chunk_overlap=50,  # Overlap to maintain context
            separator=". "  # Split on sentences
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} text chunks")
        
        return chunks
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create embeddings and store them in ChromaDB.
        
        Args:
            documents (List[Document]): List of document chunks to embed
        """
        print("🔍 Creating embeddings and storing in ChromaDB...")
        
        # Create or load existing vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist the database
        self.vectorstore.persist()
        print(f"💾 Vector store created and persisted to {self.persist_directory}")
    
    def setup_qa_chain(self) -> None:
        """
        Set up the RetrievalQA chain combining retriever and LLM.
        """
        print("🔗 Setting up RetrievalQA chain...")
        
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized! Call create_vector_store first.")
        
        # Create retriever from vector store
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Simple stuffing strategy
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
        
        print("✅ QA chain ready!")
    
    def initialize_system(self) -> None:
        """
        Initialize the complete RAG system.
        """
        print("🚀 Initializing AmbedkarGPT System...")
        print("=" * 50)
        
        try:
            # Load and process text
            documents = self.load_and_process_text()
            
            # Create vector store
            self.create_vector_store(documents)
            
            # Setup QA chain
            self.setup_qa_chain()
            
            print("=" * 50)
            print("✅ AmbedkarGPT system initialized successfully!")
            print("💡 You can now ask questions about Dr. Ambedkar's speech.")
            
        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            sys.exit(1)
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question and get an answer based on the speech content.
        
        Args:
            question (str): The question to ask
            
        Returns:
            dict: Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized! Call initialize_system first.")
        
        print(f"\n🤔 Question: {question}")
        print("🔍 Searching for relevant information...")
        
        # Get answer from QA chain
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def interactive_mode(self) -> None:
        """
        Start interactive Q&A session.
        """
        print("\n" + "=" * 60)
        print("🎯 AMBEDKAR GPT - INTERACTIVE Q&A SESSION")
        print("=" * 60)
        print("Ask questions about Dr. B.R. Ambedkar's speech on caste and shastras.")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                question = input("\n❓ Your question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("\n👋 Thank you for using AmbedkarGPT! Goodbye!")
                    break
                
                # Get answer
                result = self.ask_question(question)
                
                # Display answer
                print(f"\n🤖 Answer: {result['answer']}")
                
                # Optionally show source information
                if result['source_documents']:
                    print(f"\n📚 Based on {len(result['source_documents'])} relevant text segments.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing question: {str(e)}")


def main():
    """
    Main function to run the AmbedkarGPT system.
    """
    print("🙏 Welcome to AmbedkarGPT!")
    print("A Q&A system based on Dr. B.R. Ambedkar's speech on caste and social reform.")
    
    # Check if Ollama is available
    try:
        # Test Ollama connection
        test_llm = Ollama(model="mistral:7b-instruct-q4_K_M")
        test_response = test_llm("Hello")
        print("✅ Ollama connection successful!")
    except Exception as e:
        print("❌ Error connecting to Ollama!")
        print(f"Details: {str(e)}")
        print("\nFirst, pull the quantized model:")
        print('  & "C:\\Users\\$env:USERNAME\\AppData\\Local\\Programs\\Ollama\\ollama.exe" pull mistral:7b-instruct-q4_K_M')
        print("\nThen start Ollama:")
        print('  & "C:\\Users\\$env:USERNAME\\AppData\\Local\\Programs\\Ollama\\ollama.exe" serve')
        sys.exit(1)
    
    # Initialize the system
    ambedkar_gpt = AmbedkarGPT()
    ambedkar_gpt.initialize_system()
    
    # Start interactive session
    ambedkar_gpt.interactive_mode()


if __name__ == "__main__":
    main()
