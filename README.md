# Deepfake Detection using Xception (PyTorch + Streamlit)

This project is a deepfake image detection system built using a custom fine-tuned **Xception** model. It allows users to upload an image and get predictions via a Streamlit-based web interface.

> **Note:** This project is a simplified and modified version of [ameencaslam/deepfake-detection-project-v4](https://github.com/ameencaslam/deepfake-detection-project-v4) for educational and demonstrative purposes.

---

## 🔍 Project Overview

Deepfakes use AI to manipulate facial imagery and videos. With the rise of misinformation, detecting deepfakes is critical. This project uses the **Xception** neural network to classify whether an input image is **real** or **fake**.

Technologies used:

* **PyTorch** for deep learning model
* **timm** for pretrained Xception backbone
* **Streamlit** for interactive UI
* **OpenCV**, **NumPy**, **Matplotlib** for preprocessing and plotting
* **MLflow** (optional) for model tracking

---

## 📁 Folder Structure

```
deepfake-detector-xception/
├── app.py                   # Streamlit app
├── model.py                 # DeepfakeXception model
├── train_xception.py        # Model training pipeline
├── data_handler.py          # Dataloaders (simplified)
├── requirements.txt         # Project dependencies
├── xception_converted.pth   # Pretrained model (not uploaded)
└── README.md                # Documentation (this file)
```

---

## 🧠 How the Project Works

1. **Model Architecture:**

   * We use `timm` to load a pretrained **Xception** model (feature extractor).
   * A custom classifier head is added to perform binary classification (real/fake).

2. **Training:**

   * Images are resized to `299x299` (Xception requirement).
   * Training is done using BCEWithLogitsLoss.
   * The dataset is split into training, validation, and test sets.

3. **Prediction App:**

   * The trained model is loaded in `app.py`.
   * Users can upload an image.
   * Image is preprocessed and passed through the model.
   * A probability score is shown along with the prediction.

---

## 🔧 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/deepfake-detector-xception.git
cd deepfake-detector-xception
```

### 2. Create Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Pretrained Model

Place the pretrained weights file `xception_converted.pth` in the project directory.
(You can download it from the [original repo](https://www.kaggle.com/datasets/ameencaslam/ddp-v4-models) or train your own.)

---

## 🚀 Run the App

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

The best model will be saved as `xception_converted.pth`.

---

## 🧪 Example

* Upload an image (JPEG or PNG).
* Get prediction:

```
Probability of being Deepfake: 0.0001
✅ This image is likely Real
```

---

## ✅ requirements.txt (Summary)

```
torch
timm
opencv-python
streamlit
pandas
numpy
matplotlib
scikit-learn
mlflow
seaborn
tqdm
```


