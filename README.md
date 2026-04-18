🤖 RAG Chatbot with PDF Support

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDFs and ask questions from them, or chat normally using a local LLM.

🚀 Features
📂 Upload and chat with multiple PDFs
🔍 Semantic search using embeddings
🧠 Context-aware answers (RAG pipeline)
💬 Chat history support
⚡ Fast retrieval with ChromaDB
🤖 Local LLM using Ollama (Mistral)
🧹 Clear chat & document storage
🛠️ Tech Stack
Python
Streamlit
LangChain
ChromaDB (Vector Database)
Sentence Transformers (Embeddings)
Ollama (Mistral LLM)
PyMuPDF (fitz)
📁 Project Structure
.
├── app.py
├── uploaded_docs/       # Stores uploaded PDFs
├── chroma_db/           # Vector DB storage
├── requirements.txt
└── README.md
⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
2️⃣ Create virtual environment
python -m venv venv

Activate it:

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Install Ollama (Required)

Download from: https://ollama.com

Pull the model:

ollama pull mistral

Run the model:

ollama run mistral
▶️ Run the App
streamlit run app.py
🧠 How It Works
📄 Extracts text from PDFs using PyMuPDF
✂️ Splits text into chunks
🔢 Converts chunks into embeddings
🗄️ Stores embeddings in ChromaDB
🔍 Retrieves relevant chunks for a query
🤖 Sends context + query to LLM (Mistral)
💬 Returns accurate answers
🔄 Modes
🟢 RAG Mode (With PDFs)
Uses document context
Displays source files
🔵 Chat Mode (Without PDFs)
Acts like a general chatbot
🧹 Controls
📂 Upload PDFs → Add documents
🗑️ Clear PDFs → Remove all data
🔄 Clear Chat → Reset chat history
📸 Demo

Add your screenshots or demo video link here

🎯 Future Improvements
🔐 User authentication
☁️ Cloud deployment
⚡ Streaming responses
📊 Improved UI/UX
📄 Support for more file types
🤝 Contributing

Contributions are welcome!
Feel free to fork and submit a PR.
