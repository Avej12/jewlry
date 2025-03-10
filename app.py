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
import boto3
from io import BytesIO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# AWS S3 Configuration
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")
S3_REGION = 'ap-south-1'
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize S3 Client
s3_client = boto3.client("s3", region_name=S3_REGION, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CLIP model setup
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

UPLOAD_FOLDER = 'static/uploads'
DATA_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

CATEGORIES = {
    "rings": "datasets/All_Rings/",
    "bangles": "datasets/Bangles_new/",
    "noserings": "datasets/nose/",
    "earrings": "datasets/Unique_Earrings/",
    "mangalsutras": "datasets/Mangalsutra/"
}

indexes = {}
features_dict = {}
image_paths_dict = {}

def load_images_from_s3(category):
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=CATEGORIES[category])
        if 'Contents' not in response:
            logger.warning(f"No images found in S3 for category {category}.")
            return []
        image_keys = [obj['Key'] for obj in response['Contents'] if obj['Key'].lower().endswith(('jpg', 'jpeg', 'png', 'webp'))]
        logger.info(f"Loaded {len(image_keys)} images from S3 for {category}")
        return image_keys
    except Exception as e:
        logger.error(f"Error loading images from S3: {e}")
        return []

def fetch_image_from_s3(image_key):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=image_key)
        return Image.open(BytesIO(response['Body'].read())).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to fetch image {image_key} from S3: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    return image.resize(target_size)

def extract_clip_features(image):
    image = preprocess_image(image)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.squeeze().cpu().numpy()


def build_faiss_index(image_keys, category):
    if not image_keys:
        logger.error(f"No images found for category {category}, skipping index creation.")
        return None, None
    
    features = []
    valid_images = []
    for key in image_keys:
        img = fetch_image_from_s3(key)
        if img is not None:
            features.append(extract_clip_features(img))
            valid_images.append(key)
    
    features = np.array(features, dtype=np.float32)
    pickle.dump(features, open(os.path.join(DATA_FOLDER, f"{category}_features.pkl"), "wb"))
    pickle.dump(valid_images, open(os.path.join(DATA_FOLDER, f"{category}_image_paths.pkl"), "wb"))
    
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
        features = pickle.load(open(features_path, "rb"))
        image_paths = pickle.load(open(image_paths_path, "rb"))
        return index, features, image_paths
    return None, None, None

for category in CATEGORIES:
    index, features, image_paths = load_faiss_index(category)
    if index is None:
        image_paths = load_images_from_s3(category)
        index, features = build_faiss_index(image_paths, category)
    indexes[category] = index
    features_dict[category] = features
    image_paths_dict[category] = image_paths

def get_similar_images(query_image, category, top_k=5):
    if category not in indexes or indexes[category] is None:
        logger.error(f"No valid index for category {category}")
        return []

    query_embedding = extract_clip_features(query_image).astype(np.float32).reshape(1, -1)
    distances, indices = indexes[category].search(query_embedding, top_k + 1)  # Retrieve one extra result

    recommended_images = [
        image_paths_dict[category][i] for i in indices[0][1:top_k+1]  # Skip the first result
        if i < len(image_paths_dict[category])
    ]
    return recommended_images


def image_to_base64(img):
    try:
        if isinstance(img, Image.Image):  # If image is a PIL image
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(img, str) and os.path.exists(img):  # If image is a file path
            with open(img, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        else:
            logger.error(f"Invalid image input for base64 conversion: {img}")
            return None
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    category = request.form.get('category')
    file = request.files.get('image')

    if not category or category not in CATEGORIES or not file:
        return jsonify({'error': 'Invalid input'}), 400

    query_image = Image.open(file).convert("RGB")
    query_image = preprocess_image(query_image)  # Resize to lower resolution
    recommended_images = get_similar_images(query_image, category)

    # Optionally, encode and return the query image as well (if needed)
    query_buffer = BytesIO()
    query_image.save(query_buffer, format="JPEG")
    query_base64 = base64.b64encode(query_buffer.getvalue()).decode('utf-8')

    return jsonify({
        'query_image': query_base64,
        'recommended_images': recommended_images
    })


if __name__ == '__main__':
    app.run(debug=True)
