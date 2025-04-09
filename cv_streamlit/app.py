import variational_autoencoder
import classifier
import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ",device)

class_name = [
    'Tomato__Target_Spot',                            # 0
    'Tomato_Early_blight',                            # 1
    'Tomato_Leaf_Mold',                               # 2
    'Tomato_Bacterial_spot',                          # 3
    'Potato___Early_blight',                          # 4
    'Tomato_Spider_mites_Two_spotted_spider_mite',    # 5
    'Tomato_Septoria_leaf_spot',                      # 6
    'Tomato_healthy',                                 # 7
    'Potato___healthy',                               # 8
    'Tomato__Tomato_mosaic_virus',                    # 9
    'Tomato__Tomato_YellowLeaf__Curl_Virus',          # 10
    'Pepper__bell___healthy',                         # 11
    'Potato___Late_blight',                           # 12
    'Pepper__bell___Bacterial_spot',                  # 13
    'Tomato_Late_blight'                              # 14
]

image_path = "/content/tomato_bacteria_spot.jpg"
img = Image.open(image_path).convert("RGB")

# Load weights
variational_autoencoder.load_state_dict(torch.load('cv_streamlit/vae_weights.pth'))
classifier.load_state_dict(torch.load('cv_streamlit/classifier_vae_weights.pth'))

# Set both to eval mode
variational_autoencoder.eval()
classifier.eval()

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
input_image = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 64, 64]

# Get the 30-dimensional encoded vector
with torch.no_grad():
    encoded_feature, _, _, _ = variational_autoencoder(input_image)

# Classify using the classifier
with torch.no_grad():
    output = classifier(encoded_feature)
    predicted_class = torch.argmax(output, dim=1).item()

print("Predicted Class Index:", predicted_class)
print("Predicted Class Label:", class_name[predicted_class])
