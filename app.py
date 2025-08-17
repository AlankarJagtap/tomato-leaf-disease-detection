import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the model
model = load_model("VGG16.h5")
class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot',
               'Two Spotted Spider Mite', 'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy']

# Language selection
lang = st.sidebar.selectbox("🌐 Language / भाषा चुनें", ("English", "हिन्दी"))

# UI text translations with emojis
ui_text = {
    "English": {
        "title": "🌿 Crop Disease Classification",
        "upload": "📤 Upload an image",
        "caption": "🖼️ Uploaded Image",
        "predict": "🔍 Prediction",
        "no_file": "⚠️ Please upload an image to classify."
    },
    "हिन्दी": {
        "title": "🌿 फ़सल रोग वर्गीकरण",
        "upload": "📤 एक छवि अपलोड करें",
        "caption": "🖼️ अपलोड की गई छवि",
        "predict": "🔍 पूर्वानुमान",
        "no_file": "⚠️ कृपया वर्गीकृत करने के लिए एक छवि अपलोड करें।"
    }
}

# Disease info with emojis
disease_info = {
    "English": {
        "Bacterial Spot": "Caused by bacteria, appears as water-soaked spots. Control with copper-based sprays.",
        "Early Blight": "Fungal disease causing concentric rings on leaves. Remove infected plants and apply fungicide.",
        "Late Blight": "Serious fungal disease. Leaves show dark blotches. Use resistant varieties and fungicides.",
        "Leaf Mold": "Fungus causing yellow spots on upper leaf surfaces. Improve air circulation and apply fungicide.",
        "Septoria Leaf Spot": "Small dark spots with tan centers. Remove debris and spray with fungicide.",
        "Two Spotted Spider Mite": "Tiny pests causing speckled leaves. Use insecticidal soap or neem oil.",
        "Target Spot": "Dark brown spots with concentric rings. Apply fungicides and avoid overhead watering.",
        "Yellow Leaf Curl Virus": "Viral disease spread by whiteflies. Use resistant plants and control whiteflies.",
        "Mosaic Virus": "Causes mottled patterns on leaves. Remove infected plants and control aphids.",
        "Healthy": "The plant appears to be healthy. No signs of disease detected."
    },
    "हिन्दी": {
        "Bacterial Spot": "यह जीवाणुओं के कारण होता है और पत्तियों पर गीले धब्बे दिखते हैं। कॉपर स्प्रे से नियंत्रित करें।",
        "Early Blight": "फफूंदी के कारण पत्तियों पर गोल छल्ले बनते हैं। संक्रमित पौधों को हटाएँ और फफूंदनाशी छिड़कें।",
        "Late Blight": "गंभीर फफूंद रोग। पत्तियों पर गहरे धब्बे बनते हैं। प्रतिरोधी किस्में लगाएँ और फफूंदनाशी का प्रयोग करें।",
        "Leaf Mold": "फफूंद के कारण पत्तियों पर पीले धब्बे बनते हैं। वायु संचार बढ़ाएँ और फफूंदनाशी छिड़कें।",
        "Septoria Leaf Spot": "छोटे काले धब्बे जिनका केंद्र हल्का होता है। मलबा हटाएँ और फफूंदनाशी का प्रयोग करें।",
        "Two Spotted Spider Mite": "छोटे कीट जो पत्तियों को चिह्नित करते हैं। नीम तेल या कीटनाशक साबुन का उपयोग करें।",
        "Target Spot": "गहरे भूरे धब्बे जिनमें गोल छल्ले होते हैं। फफूंदनाशी का प्रयोग करें और ऊपर से पानी देना बंद करें।",
        "Yellow Leaf Curl Virus": "यह वायरस सफेद मक्खियों से फैलता है। प्रतिरोधी पौधे लगाएँ और सफेद मक्खियों को नियंत्रित करें।",
        "Mosaic Virus": "पत्तियों पर जाल जैसे धब्बे। संक्रमित पौधे हटाएँ और कीट नियंत्रण करें।",
        "Healthy": "पौधा स्वस्थ है। कोई रोग नहीं पाया गया।"
    }
}

# Page title
st.title(ui_text[lang]["title"])

# Upload image
uploaded_file = st.file_uploader(ui_text[lang]["upload"], type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=ui_text[lang]["caption"], width=250)

    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.markdown(f"### {ui_text[lang]['predict']}: {predicted_class} ({confidence:.2f}%)")
    st.markdown(f"**{disease_info[lang][predicted_class]}**")
else:
    st.warning(ui_text[lang]["no_file"])

