import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Setup Gemini Embedding Function
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = "models/gemini-embedding-001"

    def __call__(self, input):
        # input can be a list of strings or a single string
        if isinstance(input, str):
            input = [input]
        
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings

    def name(self):
        return "gemini"

def import_products():
    # 1. Load the CSV
    print("Loading products.csv...")
    df = pd.read_csv('products.csv')
    
    # Fill NaN values to avoid errors
    df = df.fillna('')
    
    # 2. Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 3. Setup Embedding Function
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("Using Gemini embeddings...")
        emb_fn = GeminiEmbeddingFunction(api_key=gemini_key)
    else:
        print("GEMINI_API_KEY not found. Using default Chroma embeddings...")
        emb_fn = embedding_functions.DefaultEmbeddingFunction()

    # 4. Create or Get Collection 
    # Use a new collection for Gemini to avoid dimension mismatch if old data exists
    collection_name = "product_collection_gemini" if gemini_key else "product_collection"
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn
    )

    # 5. Prepare Data for Chroma
    # We'll use the 'Description' and 'Title' as the text to embed
    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # Combine title and description for better search context
        combined_text = f"Title: {row['Title']}\nDescription: {row['Description']}"
        documents.append(combined_text)
        
        # Store other useful info in metadata
        # Metadata is extra information stored with vector.
        metadatas.append({
            "title": str(row['Title']),
            "price": str(row['Price']),
            "url": str(row['URL']),
            "image_url": str(row['Image URL']),
            "category": str(row['Category'])
        })
        
        # Use ID from CSV or index
        ids.append(str(row['ID']) if row['ID'] else str(index))

    # 6. Add to Collection in batches (Chroma handles large batches well, but let's be safe)
    print(f"Adding {len(documents)} products to ChromaDB...")
    
    # Add in batches of 100 to avoid any potential limits
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Imported batch {i//batch_size + 1}")

    print("Import completed successfully!")

if __name__ == "__main__":
    import_products()

