import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load a pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Define a transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define your extract_features function here
def extract_features(image_path):
   
    image = Image.open(image_path)

    # Apply the transformations
    image = transform(image).unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():
        features = model(image)

    return features.squeeze(0)