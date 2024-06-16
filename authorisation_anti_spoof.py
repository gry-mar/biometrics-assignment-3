import ssl
import urllib.request
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn import functional as F
# %load_ext autoreload

# %autoreload 2

ssl._create_default_https_context = ssl._create_unverified_context
from src.anti_spoof.fas import flip_it

from deepface import DeepFace
import re

import streamlit as st

# model = flip_it()
# model_path = "pretrained_models/casia_flip_mcl.pth.tar"
# checkpoint = torch.load(model_path, map_location="cpu")
# model.load_state_dict(checkpoint["state_dict"], strict=False)
# model.eval()

# preprocess = transforms.Compose(
# [
# transforms.Resize([224, 224]),
# transforms.ToTensor(),
# transforms.Normalize(mean=[0.485], std=[0.229])
# ]
# )

def authorise_user(image_path):
    print('\n-------------- Authorisation --------------\n')

    # anti spoof
    # img = Image.open(image_path)

    # input = preprocess(img).unsqueeze(0)
    # cls_out, feature = model(input, norm_flag=True)
    # prob = F.softmax(cls_out, dim=1).cpu().data.numpy()

    # spoof_result = ((prob[:, 1])>= 0.5).astype(int)[0]
    face_objs = DeepFace.extract_faces(
    img_path=image_path,
    anti_spoofing = True
    )

    if not all(face_obj["is_real"] is True for face_obj in face_objs):
        return 'Authorisation DENIED, spoof detected'

    else:
        result = DeepFace.find(image_path, db_path='./database')

        distance = result[0]['distance'][0]
        person = ' '.join((result[0]['identity'][0]).split('\\')[-1].split('_')[:2])
        max_distance = 0.5

        print(f'Distance: {distance}')
        if distance <= max_distance:
            return f'Authorisation confirmed. Welcome {person}!'
        else:
            return 'Authorisation DENIED'
            
        # return result

# Streamlit app layout
st.title('User Authorisation System')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Adjust image display size using the width parameter
    st.image(image, caption='Uploaded Image.', use_column_width=False, width=250)
    st.write("")
    if st.button('Authorise'):
        # Save the uploaded file to a temporary file to process it
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        result = authorise_user("temp_image.jpg")
        st.write(result)