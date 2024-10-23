import torch
import clip
from PIL import Image

# Load the CLIP model and preprocess functions
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/home/hexiangyu/FreeDoM/RN50.pt", device=device)

# Load and preprocess the image
image = preprocess(Image.open("/home/hexiangyu/FreeDoM/SD_style/outputs/txt2img-samples/samples/00010.png")).unsqueeze(0).to(device)

# Prepare the text prompt
text = clip.tokenize(["a dog wearing glasses."]).to(device)

# Compute the image and text features
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# Normalize the features
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Compute the similarity (cosine similarity)
similarity = (image_features @ text_features.T).item()

print(f"Similarity score: {similarity}")