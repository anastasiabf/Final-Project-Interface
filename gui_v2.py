# import module
import streamlit as st
import os
from PIL import Image
import numpy as np 
import cv2
import io
from io import BytesIO
import time
from torchvision import transforms
from model_HM import Unet_HM
from model_MA import Unet_MA
from model_EX import Unet_EX
from model_BV import Unet_BV
import scipy.ndimage as ndi
import math
import torch
import random
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import imblearn
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE

os.environ['PYOPENGL_PLATFORM'] = 'egl'

#---------------------------------------------Preprocessing---------------------------------------
def load_image(img):
    im = Image.open(img)
    return im

def normalize_img(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # normalize the pixel intensities to be between 0 and 1
    normalized_img = gray / 255.0

    return normalized_img
    
def crop_img(image):
    # normalize the image pixel intensities
    normalized_img = normalize_img(image)

    # apply thresholding to the normalized image
    threshold_value = 0.1
    _, mask = cv2.threshold(normalized_img, threshold_value, 1, cv2.THRESH_BINARY)

    # find the coordinates of the non-zero elements in the mask
    coords = np.argwhere(mask)

    # find the smallest and largest coordinates in each dimension
    x = np.min(coords, axis=0)
    y = np.max(coords, axis=0) + 1

    # crop the image based on the smallest and largest coordinates
    cropped_img = image[x[0]:y[0], x[1]:y[1]]

    return cropped_img

def color_norm(img):
  pixels = np.array(img)
  ar_mean = np.mean(pixels, axis=(0,1)) 
  Ri = pixels[:,:,0]
  Gi = pixels[:,:,1]
  Bi = pixels[:,:,2]
  (H,W) = pixels.shape[:2]

  for x in range(H):
    for y in range(W):
      if ar_mean[0] != 0:
          val = np.min(Ri[x, y] / float(ar_mean[0]) * 124.21850142079624)
          Ri[x, y] = 255 if val > 255 else val
      if ar_mean[1] != 0:
          val = np.min(Gi[x, y] / float(ar_mean[1]) * 61.74248535662327)
          Gi[x, y] = 255 if val > 255 else val
      if ar_mean[2] != 0:
          val = np.min(Bi[x, y] / float(ar_mean[2]) * 15.596426572394947)
          Bi[x, y] = 255 if val > 255 else val
    
  merged = np.dstack((Ri,Gi,Bi))
  return merged

def contrast_enhance(img):
  clahe = cv2.createCLAHE(clipLimit= 1.0, tileGridSize=(15, 15))

  Red = img[...,0]
  Green = img[...,1]
  Blue = img[...,2]

  Green_fix = clahe.apply(Green)
  new_img = np.stack([Red, Green_fix, Blue], axis=2)
  return new_img

def img_preprocess(img):
    img = cv2.resize(img, (512,512))
    img = color_norm(img)
    img = contrast_enhance(img)
    img = cv2.medianBlur(img,3)
    return img

def save_image(image_bytes, path):
    with open(path, 'wb') as f:
        f.write(image_bytes)

#------------------------------Segmentation-----------------------------------
def process_segment(img):
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    transform = transforms.ToTensor()
    preprocessed_image = transform(img)
    return preprocessed_image

# Function to perform the prediction
def predict_MA(img):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet_MA()  
    model.to(device)
    
    # Load the model weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('modelMA.pth'))
    else:
        model.load_state_dict(torch.load('modelMA.pth', map_location=torch.device('cpu')))

    # Process the image
    processed_image = process_segment(img)

    # Perform the prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model(processed_image.unsqueeze(0).to(device))
        pred = torch.sigmoid(output.squeeze(0))
        pred = (pred > 0.5).float()
        return pred

def predict_HM(img):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet_HM()  
    model.to(device)
    
    # Load the model weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('modelHM.pth'))
    else:
        model.load_state_dict(torch.load('modelHM.pth', map_location=torch.device('cpu')))

    # Process the image
    processed_image = process_segment(img)

    # Perform the prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model(processed_image.unsqueeze(0).to(device))
        pred = torch.sigmoid(output.squeeze(0))
        pred = torch.round(pred)
        return pred

def predict_EX(img):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet_EX()  
    model.to(device)
    
    # Load the model weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('modelEX.pth'))
    else:
        model.load_state_dict(torch.load('modelEX.pth', map_location=torch.device('cpu')))

    # Process the image
    processed_image = process_segment(img)

    # Perform the prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model(processed_image.unsqueeze(0).to(device))
        pred = torch.sigmoid(output.squeeze(0))
        pred = (pred > 0.5).float()
        return pred

def predict_BV(img):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet_BV()  
    model.to(device)
    
    # Load the model weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('modelBV.pth'))
    else:
        model.load_state_dict(torch.load('modelBV.pth', map_location=torch.device('cpu')))

    # Process the image
    processed_image = process_segment(img)

    # Perform the prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model(processed_image.unsqueeze(0).to(device))
        pred = torch.sigmoid(output.squeeze(0))
        pred = (pred > 0.5).float()
        return pred

#------------------------------Feature Extraction-----------------------------------
def min_axis(img):
    label_img = label(img)
    regions = regionprops(label_img)
    
    # Array initialization to save results
    min_axes = []

    # Loop through each region
    for region in regions:
        minor_axis = region.minor_axis_length

        # add result to array
        min_axes.append(minor_axis)

    if len(min_axes) > 0:
        mean_minor_axis = np.mean(min_axes)
    else:
        mean_minor_axis = 0
    
    return mean_minor_axis

def maj_axis(img):
    label_img = label(img)
    regions = regionprops(label_img)
    
    maj_axes = []

    for region in regions:
        major_axis = region.major_axis_length

        maj_axes.append(major_axis)

    if len(maj_axes) > 0:
        mean_major_axis = np.mean(maj_axes)
    else:
        mean_major_axis = 0
    
    return mean_major_axis

def calc_perimeter(img):
    # Apply thresholding for edge detection
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find lesion contour
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate perimeter for each lesion contour
    perimeters = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        perimeters.append(perimeter)

    # Calculate mean perimeter
    if len(perimeters) > 0:
        avg_perimeter = np.mean(perimeters)
    else:
        avg_perimeter = 0

    return avg_perimeter

def calc_object(img):
  labels, nlabels = ndi.label(img)
  return nlabels

#------------------------------------------------Load MLSVM Classification Model------------------------------------------
# Load model from file
with open('modelSVMCC.pkl', 'rb') as file:
    loaded_model_svm = pickle.load(file)

#------------------------------------------------START GUI HERE------------------------------------------
#----------------------------------------------------COVER--------------------------------------------
def Cover():
    # logo
    logo = Image.open("logo.png")
    st.image(logo, width = 100)
    # Title
    st.title("📙Tugas Akhir")
    # Header
    st.header("Klasifikasi Tingkat Keparahan Retinopati Diabetik dari Citra Fundus Retina Menggunakan Multi Label _Support Vector Machine_")
    # Text
    st.subheader("Anastasia Berlianna F - 0731194000004")
    st.markdown("### Dosen Pembimbing 1 : Dr. Tri Arief Sardjono, S.T., M.T\n"
                "### Dosen Pembimbing 2 : Nada Fitrieyatul Hikmah, S.T., M.T")
    # inf0
    st.sidebar.info("Please note the information below:\n"
                    "- Select **🟣PREPROCESSING** page to enhancing image quality\n"
                    "- Select **🟢PREDICTION** page to predict diabetic retinopathy severity using **:green[preprocessed image]** \n"
                    "- Select **🟡ABOUT** page to find more information")

def Preprocessing():
    # logo
    logo = Image.open("logo.png")
    st.image(logo, width = 100)

    # initialization session
    if "im" not in st.session_state:
        st.session_state["im"] = None
    if "img_crop" not in st.session_state:
        st.session_state["img_crop"] = None
    if "img_resize" not in st.session_state:
        st.session_state["img_resize"] = None
    if "img_color" not in st.session_state:
        st.session_state["img_color"] = None
    if "img_clahe" not in st.session_state:
        st.session_state["img_clahe"] = None
    if "img_medfilt" not in st.session_state:
        st.session_state["img_medfilt"] = None

    # button reset
    st.sidebar.markdown('Press **:red[RESET]** button to reset this page')
    reset = st.sidebar.button("Reset", use_container_width=True)
    if reset:
        st.session_state["im"]      = None
        st.session_state["img_crop"] = None
        st.session_state["img_resize"] = None
        st.session_state["img_color"] = None
        st.session_state["img_clahe"] = None
        st.session_state["img_medfilt"] = None
        st.sidebar.success("This page has been reset.")

    # upload single image
    st.markdown("# Image Preprocessing")
    img = st.sidebar.file_uploader(label="Upload an image", type=['jpg', 'png'])

    if img is not None:
        st.session_state["im"] = load_image(img)
        width, height = st.session_state["im"].size
        st.sidebar.write("original image size : {} x {}".format(width, height))
        st.markdown("### Original Image")
        st.image(st.session_state["im"], width = 250)

    col1, col2 = st.columns(2)

    st.sidebar.title("Preprocessing Steps")
    # button shape normalization
    crop = st.sidebar.button("Start Preprocessing", use_container_width=True)
    if crop:
        img_np = np.array(Image.open(img))
        st.session_state["img_crop"] = crop_img(img_np)
        st.session_state["img_resize"] = cv2.resize(st.session_state["img_crop"], (512,512))
        st.session_state["img_color"] = color_norm(st.session_state["img_resize"])
        st.session_state["img_clahe"] = contrast_enhance(st.session_state["img_color"])
        st.session_state["img_medfilt"] = cv2.medianBlur(st.session_state["img_clahe"],3)

    if st.session_state["img_crop"] is not None:
        with col1:
            st.markdown("### Shape Normalized Image")
            st.image(st.session_state["img_crop"], use_column_width = True)

    if st.session_state["img_resize"] is not None:
        w, h = st.session_state["img_resize"].shape[:2]
        st.sidebar.write("preprocessed image size : {} x {}".format(w, h))
        with col2:
            st.markdown("### Resized Image")
            st.image(st.session_state["img_resize"], use_column_width = True)

    if st.session_state["img_color"] is not None:
        with col1:
            st.markdown("### Color Normalized Image")
            st.image(st.session_state["img_color"], use_column_width = True)
        
    if st.session_state["img_clahe"] is not None:
        with col2:
            st.markdown("### Contrast Enhanced Image")
            st.image(st.session_state["img_clahe"], use_column_width = True)

    if st.session_state["img_medfilt"] is not None:
        with col1:
            st.markdown("### Median Filtered Image")
            st.image(st.session_state["img_medfilt"], use_column_width = True) 
   
    # preprocessing info
    st.sidebar.info("ℹ️ When pressing the button above, a series of preprocessing steps will be carried out\n"
                    "- 1️⃣Shape Normalization\n"
                    "- 2️⃣Resize\n"
                    "- 3️⃣Color Normalization\n"
                    "- 4️⃣Contrast Enhancement\n"
                    "- 5️⃣Median Filter")

    # button download result
    if st.session_state["img_medfilt"] is not None:
        im_pil = Image.fromarray(st.session_state["img_medfilt"])
        im_bytes = BytesIO()
        im_pil.save(im_bytes, format='JPEG')
        file_name = img.name.split('.')[0] + '_preproc.jpg'
        download_path = os.path.join(os.path.dirname(img.name), file_name)  # Menggabungkan direktori kerja saat ini dengan nama file yang baru
        st.sidebar.download_button(
            label="Download Image",
            data=im_bytes.getvalue(),
            file_name=file_name,
            mime="image/jpeg",
            use_container_width=True,
            on_click=lambda: save_image(im_bytes.getvalue(), download_path)
            )
        
#---------------------------------------------START PREDICTION HERE-----------------------------------
def Prediksi():
    # logo
    logo = Image.open("logo.png")
    st.image(logo, width = 100)

    # initialization session
    if "im_aft" not in st.session_state:
        st.session_state["im_aft"] = None
    if "img_MA" not in st.session_state:
        st.session_state["img_MA"] = None
    if "img_HM" not in st.session_state:
        st.session_state["img_HM"] = None
    if "img_EX" not in st.session_state:
        st.session_state["img_EX"] = None
    if "img_BV" not in st.session_state:
        st.session_state["img_BV"] = None
    if "maj_ax_MA" not in st.session_state:
        st.session_state["maj_ax_MA"] = None
    if "perimeter_MA" not in st.session_state:
        st.session_state["perimeter_MA"] = None
    if "min_ax_HM" not in st.session_state:
        st.session_state["min_ax_HM"] = None
    if "maj_ax_HM" not in st.session_state:
        st.session_state["maj_ax_HM"] = None
    if "perimeter_HM" not in st.session_state:
        st.session_state["perimeter_HM"] = None
    if "obj_HM" not in st.session_state:
        st.session_state["obj_HM"] = None
    if "min_ax_EX" not in st.session_state:
        st.session_state["min_ax_EX"] = None
    if "maj_ax_EX" not in st.session_state:
        st.session_state["maj_ax_EX"] = None
    if "perimeter_EX" not in st.session_state:
        st.session_state["perimeter_EX"] = None
    if "prediction_label" not in st.session_state:
        st.session_state["prediction_label"] = None

    st.markdown("# Diabetic Retinopathy Severity Prediction")
    # button reset
    st.sidebar.markdown('Press **:red[RESET]** button to reset this page')
    reset = st.sidebar.button("Reset", use_container_width=True)
    if reset:
        st.session_state["im_aft"]          = None
        st.session_state["img_MA"]          = None
        st.session_state["img_HM"]          = None
        st.session_state["img_EX"]          = None
        st.session_state["img_BV"]          = None
        st.session_state["maj_ax_MA"]       = None
        st.session_state["perimeter_MA"]    = None
        st.session_state["min_ax_HM"]       = None
        st.session_state["maj_ax_HM"]       = None
        st.session_state["perimeter_HM"]    = None
        st.session_state["obj_HM"]          = None
        st.session_state["min_ax_EX"]       = None
        st.session_state["maj_ax_EX"]       = None
        st.session_state["perimeter_EX"]    = None
        st.session_state["prediction_label"]= None
        st.sidebar.success("This page has been reset.")

    img = st.sidebar.file_uploader(label = "Please Upload Preprocessed Image", type=['jpg', 'png'])
    if img is not None:
        st.session_state["im_aft"] = load_image(img)
        st.markdown("### Preprocessed Image")
        st.image(st.session_state["im_aft"], width = 250)

    col1, col2 = st.columns(2)
    st.sidebar.title("Start Prediction Here")

    #button MA segmentation
    segment_MA = st.sidebar.button("Segmentasi Mikroaneurisma", use_container_width=True)
    if segment_MA:
        with st.spinner("Segmentation in progress..."):
            processed_image_MA = process_segment(st.session_state["im_aft"])
            prediction_MA = predict_MA(processed_image_MA)
            prediction_array_MA = prediction_MA.squeeze().cpu().numpy()
            prediction_image_MA = (prediction_array_MA * 255).astype(np.uint8)
            st.session_state["img_MA"] = Image.fromarray(prediction_image_MA)

        # Stop spinner
        st.success("Segmentation Done!")

    if st.session_state["img_MA"] is not None:
        with col1:
            st.markdown("### Lesi Mikroaneurisma")
            st.image(st.session_state["img_MA"], use_column_width = True)

    # button HM segmentation
    segment_HM = st.sidebar.button("Segmentasi Hemorrhages", use_container_width=True)
    if segment_HM:
        with st.spinner("Segmentation in progress..."):
            processed_image_HM = process_segment(st.session_state["im_aft"])
            prediction_HM = predict_HM(processed_image_HM)
            prediction_array_HM = prediction_HM.squeeze().cpu().numpy()
            prediction_image_HM = (prediction_array_HM * 255).astype(np.uint8)
            st.session_state["img_HM"] = Image.fromarray(prediction_image_HM)

        # Stop spinner
        st.success("Segmentation Done!")

    if st.session_state["img_HM"] is not None:
        with col2:
            st.markdown("### Lesi Hemorrhages")
            st.image(st.session_state["img_HM"], use_column_width = True)

    # button EX segmentation
    segment_EX = st.sidebar.button("Segmentasi Eksudat", use_container_width=True)
    if segment_EX:
        with st.spinner("Segmentation in progress..."):
            processed_image_EX = process_segment(st.session_state["im_aft"])
            prediction_EX = predict_EX(processed_image_EX)
            prediction_array_EX = prediction_EX.squeeze().cpu().numpy()
            prediction_image_EX = (prediction_array_EX * 255).astype(np.uint8)
            st.session_state["img_EX"] = Image.fromarray(prediction_image_EX)

        # Stop spinner
        st.success("Segmentation Done!")

    if st.session_state["img_EX"] is not None:
        with col1:
            st.markdown("### Lesi Eksudat")
            st.image(st.session_state["img_EX"], use_column_width = True)

    # button blood vessel segmentation
    segment_BV = st.sidebar.button("Segmentasi Pembuluh Darah", use_container_width=True)
    if segment_BV:
        with st.spinner("Segmentation in progress..."):
            processed_image_BV = process_segment(st.session_state["im_aft"])
            prediction_BV = predict_BV(processed_image_BV)
            prediction_array_BV = prediction_BV.squeeze().cpu().numpy()
            prediction_image_BV = (prediction_array_BV * 255).astype(np.uint8)
            st.session_state["img_BV"] = Image.fromarray(prediction_image_BV)

        # Stop spinner
        st.success("Segmentation Done!")

    if st.session_state["img_BV"] is not None:
        with col2:
            st.markdown("### Pembuluh Darah")
            st.image(st.session_state["img_BV"], use_column_width = True)

#-----------------------------------------------START FEATURE EXTRACTION HERE-----------------------------------
    # button feature extraction
    ekstrak = st.sidebar.button("Ekstraksi Fitur", use_container_width=True)
    if ekstrak:
        # change image to array
        img_MA = np.array(st.session_state["img_MA"])
        img_HM = np.array(st.session_state["img_HM"])
        img_EX = np.array(st.session_state["img_EX"])

        # Feature extraction MA lesion
        maj_ax_MA = maj_axis(img_MA.astype(np.uint8))
        perimeter_MA = calc_perimeter(img_MA.astype(np.uint8))

        # Feature extraction HM lesion
        min_ax_HM = min_axis(img_HM.astype(np.uint8))
        maj_ax_HM = maj_axis(img_HM.astype(np.uint8))
        perimeter_HM = calc_perimeter(img_HM.astype(np.uint8))
        obj_HM = calc_object(img_HM.astype(np.uint8))

        # Feature extraction EX lesion
        min_ax_EX = min_axis(img_EX.astype(np.uint8))
        maj_ax_EX = maj_axis(img_EX.astype(np.uint8))
        perimeter_EX = calc_perimeter(img_EX.astype(np.uint8))

        # Save feature extraction value in the session state
        st.session_state["maj_ax_MA"] = maj_ax_MA
        st.session_state["perimeter_MA"] = perimeter_MA
        st.session_state["min_ax_HM"] = min_ax_HM
        st.session_state["maj_ax_HM"] = maj_ax_HM
        st.session_state["perimeter_HM"] = perimeter_HM
        st.session_state["obj_HM"] = obj_HM
        st.session_state["min_ax_EX"] = min_ax_EX
        st.session_state["maj_ax_EX"] = maj_ax_EX
        st.session_state["perimeter_EX"] = perimeter_EX
    
    # Display feature extraction result
    if all(st.session_state.get(key) is not None for key in ["maj_ax_MA", "perimeter_MA", "min_ax_HM", "maj_ax_HM", "perimeter_HM", "obj_HM", "min_ax_EX", "maj_ax_EX", "perimeter_EX"]):
        with col1:
            st.info("Informasi Ekstraksi Fitur:\n"
                    "- Major Axis Length Lesi MA: {}\n"
                    "- Perimeter Lesi MA: {}\n"
                    "- Minor Axis Length Lesi HM: {}\n"
                    "- Major Axis Length Lesi HM: {}\n"
                    "- Jumlah Objek Lesi HM: {}\n"
                    "- Perimeter Lesi HM: {}\n"
                    "- Minor Axis Length Lesi EX: {}\n"
                    "- Major Axis Length Lesi EX: {}\n"
                    "- Perimeter Lesi EX: {}".format(st.session_state["maj_ax_MA"], st.session_state["perimeter_MA"], st.session_state["min_ax_HM"], 
                                                    st.session_state["maj_ax_HM"], st.session_state["obj_HM"], st.session_state["perimeter_HM"],
                                                    st.session_state["min_ax_EX"], st.session_state["maj_ax_EX"], st.session_state["perimeter_EX"]))


#-----------------------------------------------START CLASSIFICATION HERE-----------------------------------
    predict = st.sidebar.button("Prediksi", use_container_width=True)
    if predict:
        maj_ax_MA = st.session_state["maj_ax_MA"]
        perimeter_MA = st.session_state["perimeter_MA"]
        min_ax_HM = st.session_state["min_ax_HM"]
        maj_ax_HM = st.session_state["maj_ax_HM"]
        perimeter_HM = st.session_state["perimeter_HM"]
        obj_HM = st.session_state["obj_HM"]
        min_ax_EX = st.session_state["min_ax_EX"]
        maj_ax_EX = st.session_state["maj_ax_EX"]
        perimeter_EX = st.session_state["perimeter_EX"]

        # constructing array of feature extraction
        features = [[maj_ax_MA, perimeter_MA, min_ax_HM, maj_ax_HM, perimeter_HM, obj_HM, min_ax_EX, maj_ax_EX, perimeter_EX]]

        # mapping from number to label
        label_mapping = {0: "Normal", 1: "Mild Non Proliferative DR", 2: "Moderate Non Proliferative DR", 3: "Severe Non Proliferative DR", 4: "Proliferative Diabetic Retinopathy (PDR)"} 

        # do prediction using trained model
        prediction = loaded_model_svm.predict(features)

        # change prediction result to label
        st.session_state["prediction_label"] = label_mapping[int(prediction[0])]

    # Display prediction result
    if st.session_state["prediction_label"] is not None:
        with col2:
            if st.session_state["prediction_label"] == "Normal":
                # if result = normal, the button's color is green
                color = "green"
            else:
                # if result = abnormal, the button's color is red
                color = "red"

            st.markdown("<h2>Prediction Result:</h2>"
                        "<p style='font-size: 30px; color: {};"
                        "font-weight: bold;'>{}</p>"
                        "<br>".format(color, st.session_state["prediction_label"]),
                        unsafe_allow_html=True)

#-------------------------------------------------START ABOUT HERE--------------------------------------
def About():
    # logo
    logo = Image.open("logo.png")
    st.image(logo, width = 100)

    st.title("Klasifikasi Tingkat Keparahan Retinopati Diabetik dari Citra Fundus Retina Menggunakan Multi Label _Support Vector Machine_")
    st.info("👩‍🎓Anastasia Berlianna Febiola - 07311940000004\n"
            "- 👨‍🏫Dosen Pembimbing 1 : Dr. Tri Arief Sardjono, S.T., M.T\n"
            "- 👩‍🏫Dosen Pembimbing 2 : Nada Fitrieyatul Hikmah, S.T., M.T")

def main():
    # logo dr
    dr = Image.open("dr.png")
    st.sidebar.image(dr, width = 50)

    step = st.sidebar.selectbox("Select Page: ", ['🟠HOME', '🟣PREPROCESSING', '🟢PREDICTION', '🟡ABOUT'])
    if step == '🟠HOME':
        Cover()
    if step == '🟣PREPROCESSING':
        Preprocessing()
    if step == '🟢PREDICTION':
        Prediksi()    
    if step == '🟡ABOUT':
        About()

if __name__ == "__main__":
    main()






 
