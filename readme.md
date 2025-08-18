# 🍅 Tomato Leaf Disease Detection

<p align="center">
  <img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs41348-022-00608-5/MediaObjects/41348_2022_608_Fig1_HTML.jpg" 
       alt="Tomato Leaf Disease Example" width="600">
</p>

A **Streamlit web application** that uses deep learning models (**VGG16** and **MobileNetV2**) to classify tomato leaf images into multiple disease categories or as **healthy**.  
The app supports **English & Hindi** 🌐 and provides **disease-specific treatment suggestions**.

---

## ✨ Features

- 📤 Upload tomato leaf images (`.jpg`, `.jpeg`, `.png`)
- 🔍 Detects **10 tomato leaf diseases** + healthy leaves
- 🌐 Bilingual support: **English & हिन्दी**
- 📊 Displays **prediction confidence (%)**
- 💡 Provides **treatment & prevention tips**

---

## 🧪 Disease Classes

- 🦠 Bacterial Spot  
- 🍂 Early Blight  
- 🍁 Late Blight  
- 🌫️ Leaf Mold  
- 🔴 Septoria Leaf Spot  
- 🕷️ Two-Spotted Spider Mite  
- 🎯 Target Spot  
- 🍃 Yellow Leaf Curl Virus  
- 🧩 Mosaic Virus  
- ✅ Healthy  

---

## 🛠️ Tech Stack

- **Streamlit** – Web interface  
- **TensorFlow / Keras** – Deep learning models (VGG16, MobileNetV2)  
- **Pillow (PIL)** – Image processing  
- **NumPy** – Numerical computations  

---

## 🔽 Download Pre-trained Models

⚠️ The trained models are too large for GitHub.  
Please download them from **Google Drive**:

👉 [Download Models (VGG16.h5 & MobileNetV2.h5)](https://drive.google.com/drive/folders/1iO95xNg87nOhkJXlhI3yj2lxn2nf-2U8?usp=sharing)  

After downloading, place the files in the **project root directory**.  
Without these files, the app **cannot make predictions**.  

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/tomato-leaf-disease-detection.git
   cd tomato-leaf-disease-detection
2. **Install dependecies**
   ```bash
   pip install -r requirements.txt
1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   
📌 Future Improvements
    -🖼️ Add Grad-CAM visualizations for better explainability
    -🌱 Extend support to other crops
    -📱 Deploy as a mobile-friendly app
