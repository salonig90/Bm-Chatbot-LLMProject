from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize Gemini
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

# Setup Gemini Embedding Function
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key):
        self.model = "models/gemini-embedding-001"
        genai.configure(api_key=api_key)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            embeddings.append(result['embedding'])
        return embeddings

    def name(self):
        return "gemini"

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path="./chroma_db")

# Setup Embedding Function
if gemini_key:
    emb_fn = GeminiEmbeddingFunction(api_key=gemini_key)
else:
    emb_fn = embedding_functions.DefaultEmbeddingFunction()

# Get the collection 
collection_name = "product_collection_gemini" if gemini_key else "product_collection"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=emb_fn
)

# total product count
product_count = collection.count()
print("Total products in database:", product_count)

@app.route('/')
def home():
    total_products = collection.count()
    return render_template('index.html', total_products=total_products)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query:
        return render_template('index.html', results=[])
    
    # Get more results initially to filter them
    results = collection.query(
        query_texts=[query],
        n_results=50
    )
    
    # Filter results by distance (threshold: 0.7 is more inclusive for this model)
    threshold = 0.7
    formatted_results = []
    for i in range(len(results['documents'][0])):
        distance = results['distances'][0][i]
        # Only include if distance is within threshold (smaller distance = better match)
        if distance <= threshold:
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': distance,
                'id': results['ids'][0][i]
            })

    # If NO results meet the threshold, at least show the top 10 best matches
    if not formatted_results and results['documents'][0]:
        for i in range(min(10, len(results['documents'][0]))):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i]
            })

    total_products = collection.count()
    result_count = len(formatted_results)

    return render_template('index.html', query=query, results=formatted_results, total_products=total_products, result_count=result_count)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    if not gemini_key:
        return jsonify({'response': 'Gemini API key is not configured. Please set GEMINI_API_KEY in your environment.'})

    # 1. Search for relevant products
    # Clean up the user message to get better search results (simple keyword extraction)
    search_query = user_message.lower()
    for word in ['better', 'products', 'for', 'show', 'me', 'i', 'want', 'search', 'find', 'about', 'related', 'please', 'suggest', 'supplement', 'supplements']:
        search_query = search_query.replace(f" {word} ", " ").replace(f"{word} ", "").replace(f" {word}", "")
    
    # Search for more results and filter them by threshold
    raw_results = collection.query(
        query_texts=[search_query],
        n_results=50
    )
    
    # Use a stricter threshold (0.7) to match the frontend search behavior
    threshold = 0.7
    filtered_docs = []
    for i in range(len(raw_results['documents'][0])):
        if raw_results['distances'][0][i] <= threshold:
            filtered_docs.append(raw_results['documents'][0][i])
    
    # If no results meet the threshold, at least show the top 3 best matches (reduced from 10)
    if not filtered_docs and raw_results['documents'][0]:
        filtered_docs = raw_results['documents'][0][:3]
    
    # Cap the context at 15 products to keep the chat manageable
    final_docs = filtered_docs[:15]
    result_count = len(final_docs)

    # 2. Prepare context from search results
    context = ""
    for doc in final_docs:
        context += f"- {doc}\n"

    # 3. Generate response using Gemini
    prompt = f"""
    You are a helpful product assistant for an e-commerce store. 
    The user is asking: "{user_message}"
    
    I found {result_count} relevant products in our database. 
    Please list ALL of these {result_count} products for the user.
    
    Rules for your response:
    1. Start by saying exactly how many products you found (e.g., "I found {result_count} products for sleep:").
    2. For EACH product, provide ONLY the Title and a very brief (1-sentence) summary.
    3. Use a clear, bulleted list.
    4. Keep the descriptions concise so the user can read them quickly.
    5. Do NOT skip any products provided in the context.
    
    Context (Product List):
    {context}
    
    Answer:
    """

    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt)
        return jsonify({'response': response.text})
    except Exception as e:
        print(f"Error generating content: {str(e)}")
        return jsonify({'response': "I'm sorry, I'm having trouble connecting to the AI service right now. Please try again later."})

if __name__ == '__main__':
    app.run(debug=True, port=5001)


