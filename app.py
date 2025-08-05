import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import DeepfakeXception

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image preprocessing (same as training)
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Load model
@st.cache_resource
def load_model():
    model = DeepfakeXception().to(device)
    model.load_state_dict(torch.load("xception_converted.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("Deepfake Detection using Xception")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_file, use_container_width=True)

    # Preprocess image
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    # Display result
    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"**Probability of being Deepfake:** `{prob:.4f}`")

    if prob > 0.5:
        st.error("This image is likely a **Deepfake**.")
    else:
        st.success("This image is likely **Real**.")
