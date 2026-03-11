# AI Product Assistant & Search

A full-stack e-commerce search and chatbot application powered by **Gemini AI**, **ChromaDB**, and **Flask**. This project allows users to search for products using natural language and interact with an AI assistant to find the perfect supplements for their needs.

## 🚀 Features

- **Semantic Product Search**: Find products using natural language queries (e.g., "something for better sleep") instead of just keywords.
- **AI Chatbot Assistant**: A dedicated AI assistant that understands the product catalog and provides personalized recommendations.
- **Vector Database**: Uses ChromaDB to store and retrieve high-dimensional product embeddings.
- **Gemini Integration**: Powered by Google's Gemini-2.0-Flash for both text generation and high-quality embeddings.
- **Responsive UI**: A modern, clean interface built with Bootstrap for easy browsing.

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **AI/LLM**: Google Gemini API (`google-generativeai`)
- **Vector DB**: ChromaDB
- **Frontend**: HTML5, CSS3, JavaScript (Bootstrap 5)
- **Environment**: Python 3.10+, `python-dotenv`

## 📋 Prerequisites

Before you begin, ensure you have:
- A Google AI Studio API Key ([Get it here](https://aistudio.google.com/app/apikey))
- Python installed on your machine.

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd chatbot_miniproject
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Import Product Data**:
   Run the import script to populate the vector database with products from `products.csv`:
   ```bash
   python import_products.py
   ```

## 🏃 Running the Application

Start the Flask development server:
```bash
python app.py
```
The application will be available at `http://127.0.0.1:5001`.

## 🧪 Testing the AI

Try asking the chatbot or searching for:
- "omega supplements please suggest"
- "better products for sleep"
- "something for my immune system"
- "vitamins for hair growth"

## 📁 Project Structure

- `app.py`: The main Flask application and API routes.
- `import_products.py`: Script to process CSV data and store it in ChromaDB.
- `products.csv`: The product database.
- `templates/index.html`: The frontend user interface.
- `chroma_db/`: Persistent storage for the vector database.
- `.env`: API key configuration (git-ignored).

---
Developed as a Mini-Project for AI-Powered E-commerce Search.
