import os
import requests
from io import BytesIO
from pinecone import Pinecone
import torch
import clip
from PIL import Image
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

# Define index name and ensure it exists
index_name = os.getenv("index_name")
if index_name not in pc.list_indexes().names():
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=512,  # CLIP embedding dimension
        metric="cosine"  # Metric for similarity search
    )
else:
    print(f"Index {index_name} already exists.")

# Connect to the Pinecone index
pinecone_index = pc.Index(index_name)
print(f"Connected to index: {index_name}, Type: {type(pinecone_index)}")

# Load the CLIP model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to get image embeddings from a URL
def get_image_embeddings(image_url, model, preprocess, device):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img_data = response.content
        image = Image.open(BytesIO(img_data))
        if image.mode in ("P", "1"):
            image = image.convert("RGBA")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image).detach().cpu().numpy().flatten()
        return image_features
    except requests.RequestException as e:
        print(f"Failed to fetch image from URL: {image_url}, Error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing image URL {image_url}: {str(e)}")
        return None

# Function to get text embeddings
def get_text_embeddings(text, model, device):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).detach().cpu().numpy().flatten()
    return text_features

# Endpoint to upload and embed images and text
@app.route('/embed_chairs', methods=['POST'])
def embed_chairs():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']

    try:
        df = pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        return jsonify({"error": f"Error reading Excel file: {str(e)}"}), 400

    embedded_count = 0
    error_count = 0

    for index, row in df.iterrows():
        image_url = row.get('Thumbnail Image')
        product_name = row.get('Product Name')
        product_id = row.get('Product_id')

        if pd.isna(image_url) or pd.isna(product_name) or pd.isna(product_id):
            print(f"Skipping row {index}, invalid data.")
            error_count += 1
            continue

        print(f"Processing: {product_name} ({product_id})")
        
        image_embeddings = get_image_embeddings(image_url, model, preprocess, device)
        text_embeddings = get_text_embeddings(product_name, model, device)

        if image_embeddings is None:
            print(f"Failed to get embeddings for {product_name} ({product_id})")
            error_count += 1
            continue

        metadata = {
            "product_name": product_name,
            "product_id": product_id,
            "image_url": image_url
        }

        try:
            pinecone_index.upsert(vectors=[
                (f"{product_id}_image", image_embeddings.tolist(), metadata),
                (f"{product_id}_text", text_embeddings.tolist(), metadata)
            ], namespace="chair-image-text-namespace")
            embedded_count += 1
            print(f"Embedded: {product_name} ({product_id})")
        except Exception as e:
            print(f"Error uploading embeddings for {product_name} ({product_id}): {str(e)}")
            error_count += 1

    return jsonify({
        "message": "Chairs embedding process completed.",
        "embedded_count": embedded_count,
        "error_count": error_count
    }), 200

# Endpoint to query using either an image or text
@app.route('/query_chairs', methods=['POST'])
def query_chairs():
    data = request.get_json()
    print(f"Received query data: {data}")

    if 'image_url' not in data:
        return jsonify({"error": "No image URL provided."}), 400

    image_url = data['image_url']
    embeddings = get_image_embeddings(image_url, model, preprocess, device)
    if embeddings is None:
        return jsonify({"error": "Failed to get features for the image."}), 400
    
    try:
        exact_match = pinecone_index.query(
            vector=[0]*512,
            top_k=1,
            include_metadata=True,
            namespace="chair-image-text-namespace",
            filter={"image_url": {"$eq": image_url}}
        )
        
        similar_results = pinecone_index.query(
            vector=embeddings.tolist(), 
            top_k=5, 
            include_metadata=True, 
            namespace="chair-image-text-namespace"
        )
    except Exception as e:
        return jsonify({"error": f"Error querying Pinecone: {str(e)}"}), 500
    
    results = exact_match['matches'] + similar_results['matches']

    seen_product_ids = set()
    products = []
    for match in results:
        product_id = match['metadata']['product_id']
        if product_id not in seen_product_ids:
            seen_product_ids.add(product_id)
            products.append({
                "product_id": product_id,
                "product_name": match['metadata']['product_name'],
                "image_url": match['metadata']['image_url'],
                "score": match['score'] if 'score' in match else 1.0
            })

    return jsonify(products), 200

# Error handler for all exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    # Now handle non-HTTP exceptions
    return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5011)