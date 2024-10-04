import requests
from io import BytesIO
import chromadb
import torch
import clip
import os
from PIL import Image
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient("./db")

# Check if the collection exists, and if not, create it
collection_name = "images"
try:
    image_collection = chroma_client.get_collection(collection_name)
    print(f"Loaded existing collection: {collection_name}")
except chromadb.errors.CollectionNotFoundError:
    image_collection = chroma_client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    print(f"Created new collection: {collection_name}")

# Load the CLIP model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to get image embeddings
def get_embeddings(path, model, preprocess, device):
    try:
        response = requests.get(path)
        if response.status_code == 200:
            img_data = response.content
            image = Image.open(BytesIO(img_data))
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image).detach().cpu().numpy()
            return image_features
    except Exception as e:
        print(e)
        return None
    

# Endpoint to upload and embed images
@app.route('/embed_images', methods=['POST'])
def embed_images():
    data = request.get_json()
    excel_file = data.get('excel_file')

    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
    except Exception as e:
        return jsonify({"error": f"Error reading Excel file: {str(e)}"}), 400

    for index, row in df.iterrows():
        image_url = row.get('Thumbnail Image')
        product_name = row.get('Product Name')
        product_id = row.get('Product_id')

        if pd.isna(image_url) or pd.isna(product_name) or pd.isna(product_id):
            print(f"Skipping row {index}, invalid data.")
            continue

        print(f"Processing: {product_name} ({product_id})")
        embeddings = get_embeddings(image_url, model, preprocess, device)

        if embeddings is None:
            print(f"Failed to get embeddings for {product_name} ({product_id})")
            continue

        image_collection.add(
            ids=[str(product_id)],
            embeddings=embeddings.tolist(),
            metadatas=[{
                "path": image_url,
                "product_name": product_name,
                "product_id": str(product_id)
            }]
        )
        print(f"Embedded: {product_name} ({product_id})")

    return jsonify({"message": "Images embedded successfully."}), 200

# Endpoint to query images
@app.route('/query', methods=['POST'])
def query():
    input_file = request.files.get('input_file')  # Retrieve the image file

    if input_file is None:
        return jsonify({"error": "No image file provided."}), 400

    # Function to get embeddings for both text and image inputs
    def get_input_embeddings(input_file, model, preprocess, device):
        try:
            image = preprocess(Image.open(input_file)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image).detach().cpu().numpy()
            return image_features
        except Exception as e:
            print(f"Error processing image file: {str(e)}")
            return None

    # Get embeddings from the uploaded image file
    features = get_input_embeddings(input_file, model, preprocess, device)
    if features is None:
        return jsonify({"error": "Failed to get features for the image."}), 400
    
    # Query the image collection using the embeddings
    result = image_collection.query(query_embeddings=features.tolist(), n_results=4)

    products = []
    for metadata in result['metadatas'][0]:
        products.append({
            "product_name": metadata['product_name'],
            "product_id": metadata['product_id'],
            "image_path": metadata['path']
        })

    return jsonify(products), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
