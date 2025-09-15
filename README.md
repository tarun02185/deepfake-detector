# Deepfake Detection using Xception and EfficientNet (PyTorch + Streamlit)

This project is a deepfake image detection system built using custom fine-tuned **Xception** and **EfficientNet-B3** models. The system allows users to upload an image and get predictions via a Streamlit-based web interface.

---

## 🔍 Project Overview

Deepfakes use AI to manipulate facial imagery and videos. With the rise of misinformation, detecting deepfakes is critical. This project uses state-of-the-art deep learning architectures (**Xception** and **EfficientNet-B3**) to classify whether an input image is **real** or **fake**.

**Technologies used:**

- **PyTorch** for deep learning model development  
- **timm** and **efficientnet-pytorch** for pretrained backbones  
- **Streamlit** for interactive web UI  
- **OpenCV**, **NumPy**, **Matplotlib**, **Seaborn** for image processing and visualization  
- **MLflow** for experiment tracking and model management  

---

## 📁 Folder Structure



```
deepfake-detector/
├── app.py # Streamlit app
├── model.py # Model definitions for Xception and EfficientNet
├── efficient_net.py # EfficientNet model (optional separate file)
├── train_xception.py # Training pipeline for Xception model
├── data_handler.py # Dataset and DataLoader utilities
├── requirements.txt # Project dependencies
└── README.md # Project documentation (this file)
```


---

## 🧠 How the Project Works

1. **Model Architecture:**  
   - Uses pretrained **Xception** and **EfficientNet-B3** backbones, modified with custom classifier heads for binary classification (real/fake).  
   - Custom classification heads include batch normalization, dropout, and fully connected layers.  

2. **Training:**  
   - Images are resized per model requirements (299x299 for Xception, 300x300 for EfficientNet).  
   - Models are trained with BCEWithLogitsLoss and AdamW optimizer with separate learning rates for backbone and classifier layers.  
   - Dataset is split into training, validation, and testing sets.

3. **Prediction App:**  
   - Loads both trained models in `app.py` with caching for efficiency.  
   - Provides a Streamlit UI for image upload, preprocessing, and prediction from both models.  
   - Displays probability scores and clear, user-friendly real vs deepfake predictions.

---

## 🔧 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
```

### 2. Create Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


### 4. Add Pretrained Model Weights

Place the pretrained weights files `xception_converted.pth` and `efficientnet_converted.pth` in the project directory.  
(You can download them from the original source or train your own models following the training instructions below.)

---

## 🚀 Run the Streamlit App

```bash
streamlit run app.py
```

Then, visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔁 To Train the Model Yourself

Edit `DATA_DIR` in `train_xception.py` to point to your dataset folder.
Then run:

```bash
python train_xception.py
```
- The best model checkpoint will be saved as `xception_converted.pth`.  
- You can create a similar training script for EfficientNet or extend `train_xception.py` accordingly.

---

## 🧪 Example Usage

- Upload an image in JPG or PNG format.  
- View prediction probabilities for both models:  

```
Probability of being Deepfake (Xception): 0.0024
Probability of being Deepfake (EfficientNet-B3): 0.0031
✅ This image is likely Real
```

---

## ✅ requirements.txt (Summary)

```

## ✅ requirements.txt (Summary)

torch>=2.0.0
torchvision
timm
efficientnet-pytorch
opencv-python
streamlit
pandas
numpy
matplotlib
scikit-learn
mlflow
seaborn
tqdm
pillow
```


