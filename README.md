# AmbedkarGPT - Q&A System

A command-line Q&A system that answers questions based on Dr. B.R. Ambedkar's speech content using RAG (Retrieval-Augmented Generation) pipeline.

## 🎯 Project Overview

This system implements a complete RAG pipeline that:
1. Loads Dr. Ambedkar's speech text
2. Splits text into manageable chunks
3. Creates embeddings using HuggingFace models
4. Stores embeddings in ChromaDB vector database
5. Retrieves relevant context for user questions
6. Generates answers using Ollama Mistral 7B LLM

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **Framework**: LangChain
- **Vector Database**: ChromaDB (local, persistent)
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Ollama Mistral 7B Quantized (mistral:7b-instruct-q4_K_M) - local, free, CPU-optimized

## 📋 Prerequisites

### 1. Install Ollama
```bash
# On Windows (PowerShell as Administrator)
winget install Ollama.Ollama

# Or download from: https://ollama.ai/download
```

### 2. Pull Mistral 7B Quantized Model
```bash
ollama pull mistral:7b-instruct-q4_K_M
```

This quantized version is optimized for CPU inference and uses ~3.5GB of disk space.

### 3. Verify Ollama Installation
```bash
ollama list
# Should show mistral model
```

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv ambedkar_env

# Activate virtual environment
# On Windows:
ambedkar_env\Scripts\activate
# On macOS/Linux:
source ambedkar_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the System
```bash
python main.py
```

## 💻 Usage

### Interactive Mode
The system starts in interactive mode where you can ask questions about Dr. Ambedkar's speech:

```
🎯 AMBEDKAR GPT - INTERACTIVE Q&A SESSION
============================================================
Ask questions about Dr. B.R. Ambedkar's speech on caste and shastras.
Type 'quit', 'exit', or 'q' to end the session.
============================================================

❓ Your question: What does Ambedkar say about the shastras?

🤖 Answer: According to the text, Ambedkar argues that the real remedy is to destroy the belief in the sanctity of the shastras. He states that you cannot have both - either you must stop the practice of caste or stop believing in the shastras. He emphasizes that the real enemy is the belief in the shastras, and as long as people believe in their sanctity, they will never be able to get rid of caste.
```

### Example Questions to Try
- "What is the real remedy according to Ambedkar?"
- "What does Ambedkar say about social reform?"
- "How does Ambedkar compare social reform to gardening?"
- "What is the relationship between caste and shastras?"

## 📁 Project Structure

```
AmbedkarGPT-Intern-Task/
├── main.py              # Main application code
├── speech.txt           # Dr. Ambedkar's speech text
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── chroma_db/          # ChromaDB vector store (created on first run)
```

## 🔧 System Architecture

### RAG Pipeline Components

1. **Document Loading**: TextLoader reads the speech.txt file
2. **Text Splitting**: CharacterTextSplitter creates overlapping chunks
3. **Embeddings**: HuggingFace sentence-transformers model creates vector representations
4. **Vector Store**: ChromaDB stores and indexes embeddings locally
5. **Retrieval**: Similarity search finds relevant text chunks
6. **Generation**: Ollama Mistral 7B generates contextual answers

### Key Features

- **Local Operation**: No API keys or external services required
- **Persistent Storage**: ChromaDB saves embeddings for faster subsequent runs
- **Interactive Interface**: User-friendly command-line interaction
- **Error Handling**: Comprehensive error checking and user feedback
- **Modular Design**: Clean, well-documented code structure

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```
   Error: Ollama not responding
   Solution: Ensure Ollama is running and Mistral model is installed
   ```

2. **Import Errors**
   ```
   Error: Module not found
   Solution: Activate virtual environment and reinstall requirements
   ```

3. **File Not Found**
   ```
   Error: speech.txt not found
   Solution: Ensure speech.txt is in the same directory as main.py
   ```

### Performance Notes

- First run takes longer due to model downloads and embedding creation
- Subsequent runs are faster as ChromaDB persists the vector store
- Embedding model (~90MB) downloads automatically on first use

## 🎓 Assignment Requirements Fulfilled

✅ **Load text file**: TextLoader handles speech.txt  
✅ **Split into chunks**: CharacterTextSplitter with overlap  
✅ **Create embeddings**: HuggingFace sentence-transformers  
✅ **Vector store**: ChromaDB with local persistence  
✅ **Retrieve context**: Similarity search with top-k retrieval  
✅ **Generate answers**: Ollama Mistral 7B integration  
✅ **LangChain framework**: Complete RAG pipeline  
✅ **No API keys**: Fully local operation  
✅ **Well-commented code**: Comprehensive documentation  
✅ **Requirements.txt**: All dependencies listed  
✅ **README.md**: Complete setup and usage guide  

## 👨‍💻 Development

### Code Structure

- `AmbedkarGPT` class encapsulates the entire system
- Modular methods for each pipeline component
- Error handling and user feedback throughout
- Type hints for better code maintainability

### Extending the System

- Add more text sources by modifying `load_and_process_text()`
- Experiment with different chunk sizes and overlap
- Try different embedding models or LLMs
- Add conversation memory for multi-turn interactions

## 📝 License

This project is created for the Kalpit Pvt Ltd internship assignment.

## 📧 Contact

For questions about this implementation, please contact through the provided hiring communication channels.

---

**Built with ❤️ for Kalpit Pvt Ltd Internship Assignment**
