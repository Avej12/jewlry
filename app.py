from flask import Flask, request, render_template, jsonify
import os
from PIL import Image
import numpy as np
import torch
import faiss
import pickle
from transformers import CLIPModel, CLIPProcessor
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# CLIP model setup
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# Paths
UPLOAD_FOLDER = 'static/uploads'
DATA_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Define categories and dataset paths (Relative paths for deployment)
CATEGORIES = {
    "rings": "datasets/All_Rings",
    "bangles": "datasets/Bangles_new",
    "noserings": "datasets/nose",
    "earrings": "datasets/Unique_Earrings",
    "mangalsutras": "datasets/Mangalsutra"
}

indexes = {}
features_dict = {}
image_paths_dict = {}

def load_images(folder_path):
    if not os.path.exists(folder_path):
        logger.warning(f"Dataset folder {folder_path} does not exist!")
        return []
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    logger.info(f"Loaded {len(image_paths)} images from {folder_path}")
    return image_paths

def extract_clip_features(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.squeeze().cpu().numpy()

def build_faiss_index(image_paths, category):
    if not image_paths:
        logger.error(f"No images found for category {category}, skipping index creation.")
        return None, None
    
    features = np.array([extract_clip_features(Image.open(img).convert("RGB")) 
                        for img in image_paths], dtype=np.float32)
    
    with open(os.path.join(DATA_FOLDER, f"{category}_features.pkl"), "wb") as f:
        pickle.dump(features, f)
    with open(os.path.join(DATA_FOLDER, f"{category}_image_paths.pkl"), "wb") as f:
        pickle.dump(image_paths, f)
    
    dimension = features.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(features)
    faiss.write_index(index, os.path.join(DATA_FOLDER, f"{category}_faiss_index.bin"))
    logger.info(f"FAISS index for {category} built and saved")
    return index, features

def load_faiss_index(category):
    index_path = os.path.join(DATA_FOLDER, f"{category}_faiss_index.bin")
    features_path = os.path.join(DATA_FOLDER, f"{category}_features.pkl")
    image_paths_path = os.path.join(DATA_FOLDER, f"{category}_image_paths.pkl")
    
    if all(os.path.exists(p) for p in [index_path, features_path, image_paths_path]):
        logger.info(f"Loading FAISS index for {category}...")
        index = faiss.read_index(index_path)
        with open(features_path, "rb") as f:
            features = pickle.load(f)
        with open(image_paths_path, "rb") as f:
            image_paths = pickle.load(f)
        return index, features, image_paths
    return None, None, None



for category, folder in CATEGORIES.items():
    index, features, image_paths = load_faiss_index(category)  # Try loading existing index
    if index is not None and features is not None and image_paths is not None:
        logger.info(f"Loaded FAISS index from storage for {category}")
    else:
        logger.info(f"FAISS index for {category} not found, building it...")
        image_paths = load_images(folder)
        if not image_paths:
            continue
        index, features = build_faiss_index(image_paths, category)

    indexes[category] = index
    features_dict[category] = features
    image_paths_dict[category] = image_paths

def get_similar_images(query_image, category, top_k=6):
    if category not in indexes or indexes[category] is None:
        logger.error(f"No valid index for category {category}")
        return []
    query_embedding = extract_clip_features(query_image).astype(np.float32).reshape(1, -1)
    distances, indices = indexes[category].search(query_embedding, top_k)
    recommended_images = [image_paths_dict[category][i] for i in indices[0] if i < len(image_paths_dict[category])]
    return recommended_images[1:top_k]

def image_to_base64(img_path):
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        logger.error(f"Image not found: {img_path}")
        return None

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    category = request.form.get('category')
    if not category or category not in CATEGORIES:
        return jsonify({'error': 'Invalid category'}), 400
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    img_path = os.path.join(UPLOAD_FOLDER, 'query_image.jpg')
    file.save(img_path)
    query_image = Image.open(img_path).convert("RGB")
    
    recommended_images = get_similar_images(query_image, category)
    if not recommended_images:
        return jsonify({'error': 'No recommendations found'}), 404
    
    query_base64 = image_to_base64(img_path)
    recommended_base64 = [image_to_base64(img) for img in recommended_images]
    
    return jsonify({
        'query_image': query_base64,
        'recommended_images': recommended_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
