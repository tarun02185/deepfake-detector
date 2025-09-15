import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import DeepfakeXception, DeepfakeEfficientNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    # Xception
    xception = DeepfakeXception().to(device)
    xception.load_state_dict(torch.load("xception_converted.pth", map_location=device))
    xception.eval()

    # EfficientNet
    effnet = DeepfakeEfficientNet().to(device)
    effnet.load_state_dict(torch.load("efficientnet_converted.pth", map_location=device))
    effnet.eval()

    return xception, effnet

# Load both models once
xception_model, effnet_model = load_models()

# Preprocessing configs
preprocess_xception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

preprocess_effnet = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# UI
st.title("Deepfake Detection (Xception & EfficientNet)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_file, use_container_width=True)

    # Preprocess for each model
    input_x = preprocess_xception(image).unsqueeze(0).to(device)
    input_e = preprocess_effnet(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        prob_x = torch.sigmoid(xception_model(input_x)).item()
        prob_e = torch.sigmoid(effnet_model(input_e)).item()

    # Show results
    st.markdown("---")
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Xception")
        st.write(f"Probability of being Deepfake: `{prob_x:.4f}`")
        if prob_x > 0.5:
            st.error("Likely **Deepfake**")
        else:
            st.success("Likely **Real**")

    with col2:
        st.write("### EfficientNet-B3")
        st.write(f"Probability of being Deepfake: `{prob_e:.4f}`")
        if prob_e > 0.5:
            st.error("Likely **Deepfake**")
        else:
            st.success("Likely **Real**")
