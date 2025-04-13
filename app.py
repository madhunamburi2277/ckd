import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import shap

classes = {0: "Normal", 1: "Tumorpy"}

def load_background_batch():
    test_dir = r"test/Kidney Cancer"
    background_data = []
    
    for img_file in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_file)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        background_data.append(img_array)
    
    return np.vstack(background_data)

def predict_class(img, model):
    img = Image.open(img)
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    return predictions, img, predicted_class_idx

def shap_explanation(model, img_array, background):
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(img_array)
    return shap_values

def show_shap(shap_values, img_array, predicted_class_idx):
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    shap_values_for_predicted_class = shap_values[-1]
    
    plt.figure()
    shap.image_plot(shap_values_for_predicted_class, img_array)
    plt.show()
    
    predicted_class_name = classes[predicted_class_idx]
    st.warning(f"Model predicted: {predicted_class_name}")
    
    st.write('Inference for the Prediction: Plot of SHAP values')
    st.pyplot(plt.gcf())

def main():
    st.title('Kidney Cancer Detection App')
    
    model = load_model('./models/kidney_model.h5', compile=False)
    background_batch = load_background_batch()
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None and st.button('Predict'):
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        
        predictions, img, predicted_class_idx = predict_class(uploaded_file, model)
        shap_values = shap_explanation(model, img, background_batch)
        show_shap(shap_values, img, predicted_class_idx)

if __name__ == "__main__":
    main()