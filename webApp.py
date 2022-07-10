import streamlit as st
import streamlit.components.v1 as stc

# File Processing Pkgs
import pandas as pd
import docx2txt
from PIL import Image 
from PyPDF2 import PdfFileReader
import pdfplumber
from keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle
import h5py as h5

tfds = tf.data.Dataset



@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 



def main():
    st.title("Crop dissease classification")

    menu = ["Apple", "Cherry", "Corn", "Grape", "Peach", "Potato", "Strawberry"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Apple":

        st.subheader(choice)
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:

            # To See Details
            st.write(type(image_file))
            # st.write(dir(image_file))
            # file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            # st.write(file_details)

            img = load_image(image_file)
            st.image(img,width=480)
            st.write(image_file)

            IMG_SIZE = (1,128, 128,3)
            img_array = np.resize(np.array(img),IMG_SIZE)
            st.write(img_array.shape)
            # img_array = tf.image.resize(img_array, size=IMG_SIZE)
            # st.write(img_array)

            dataset = tf.data.Dataset.from_tensor_slices(img_array)
            # dataset = dataset.map(_parse_function)
            dataset = dataset.batch(1)
            with open(choice+'_class.pkl','rb') as clas:
                classes = pickle.load(clas)


            reconstructed_model = load_model(choice+".h5")
            out = reconstructed_model.predict(dataset)
            st.write(classes[np.argmax(out)])

    elif choice == "Cherry":

        st.subheader(choice)
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:

            # To See Details
            st.write(type(image_file))
            # st.write(dir(image_file))
            # file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            # st.write(file_details)

            img = load_image(image_file)
            st.image(img,width=480)
            st.write(image_file)

            IMG_SIZE = (1,128, 128,3)
            img_array = np.resize(np.array(img),IMG_SIZE)
            st.write(img_array.shape)
            # img_array = tf.image.resize(img_array, size=IMG_SIZE)
            # st.write(img_array)

            dataset = tf.data.Dataset.from_tensor_slices(img_array)
            # dataset = dataset.map(_parse_function)
            dataset = dataset.batch(1)

            with open(choice+'_class.pkl','rb') as clas:
                classes = pickle.load(clas)

            reconstructed_model = load_model(choice+".h5")
            out = reconstructed_model.predict(dataset)
            st.write(classes[np.argmax(out)])

    elif choice == "Peach":

        st.subheader(choice)
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:

            # To See Details
            st.write(type(image_file))
            # st.write(dir(image_file))
            # file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            # st.write(file_details)

            img = load_image(image_file)
            st.image(img,width=480)
            st.write(image_file)

            IMG_SIZE = (1,128, 128,3)
            img_array = np.resize(np.array(img),IMG_SIZE)
            st.write(img_array.shape)
            # img_array = tf.image.resize(img_array, size=IMG_SIZE)
            # st.write(img_array)

            dataset = tf.data.Dataset.from_tensor_slices(img_array)
            # dataset = dataset.map(_parse_function)
            dataset = dataset.batch(1)


            with open(choice+'_class.pkl','rb') as clas:
                classes = pickle.load(clas)


            reconstructed_model = load_model(choice+".h5")
            out = reconstructed_model.predict(dataset)
            st.write(classes[np.argmax(out)])

    elif choice == "Potato":

        st.subheader(choice)
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:

            # To See Details
            st.write(type(image_file))
            # st.write(dir(image_file))
            # file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            # st.write(file_details)

            img = load_image(image_file)
            st.image(img,width=480)
            st.write(image_file)

            IMG_SIZE = (1,128, 128,3)
            img_array = np.resize(np.array(img),IMG_SIZE)
            st.write(img_array.shape)
            # img_array = tf.image.resize(img_array, size=IMG_SIZE)
            # st.write(img_array)

            dataset = tf.data.Dataset.from_tensor_slices(img_array)
            # dataset = dataset.map(_parse_function)
            dataset = dataset.batch(1)
            with open(choice+'_class.pkl','rb') as clas:
                classes = pickle.load(clas)


            reconstructed_model = load_model(choice+".h5")
            out = reconstructed_model.predict(dataset)
            st.write(classes[np.argmax(out)])


    elif choice == "Corn":

        st.subheader(choice)
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:

            # To See Details
            st.write(type(image_file))
            # st.write(dir(image_file))
            # file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            # st.write(file_details)

            img = load_image(image_file)
            st.image(img,width=480)
            st.write(image_file)

            IMG_SIZE = (1,128, 128,3)
            img_array = np.resize(np.array(img),IMG_SIZE)
            st.write(img_array.shape)
            # img_array = tf.image.resize(img_array, size=IMG_SIZE)
            # st.write(img_array)

            dataset = tf.data.Dataset.from_tensor_slices(img_array)
            # dataset = dataset.map(_parse_function)
            dataset = dataset.batch(1)
            with open('_class.pkl','rb') as clas:
                classes = pickle.load(clas)


            reconstructed_model = load_model(choice+".h5")
            out = reconstructed_model.predict(dataset)
            st.write(classes[np.argmax(out)])


    elif choice == "Grape":

        st.subheader(choice)
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:

            # To See Details
            st.write(type(image_file))
            # st.write(dir(image_file))
            # file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            # st.write(file_details)

            img = load_image(image_file)
            st.image(img,width=480)
            st.write(image_file)

            IMG_SIZE = (1,128, 128,3)
            img_array = np.resize(np.array(img),IMG_SIZE)
            st.write(img_array.shape)
            # img_array = tf.image.resize(img_array, size=IMG_SIZE)
            # st.write(img_array)

            dataset = tf.data.Dataset.from_tensor_slices(img_array)
            # dataset = dataset.map(_parse_function)
            dataset = dataset.batch(1)
            with open(choice+'_class.pkl','rb') as clas:
                classes = pickle.load(clas)


            reconstructed_model = load_model(choice+".h5")
            out = reconstructed_model.predict(dataset)
            st.write(classes[np.argmax(out)])


    elif choice == "Strawberry":

        st.subheader(choice)
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:

            # To See Details
            st.write(type(image_file))
            # st.write(dir(image_file))
            # file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            # st.write(file_details)

            img = load_image(image_file)
            st.image(img,width=480)
            st.write(image_file)

            IMG_SIZE = (1,128, 128,3)
            img_array = np.resize(np.array(img),IMG_SIZE)
            st.write(img_array.shape)
            # img_array = tf.image.resize(img_array, size=IMG_SIZE)
            # st.write(img_array)

            dataset = tf.data.Dataset.from_tensor_slices(img_array)
            # dataset = dataset.map(_parse_function)
            dataset = dataset.batch(1)
            with open(choice+'_class.pkl','rb') as clas:
                classes = pickle.load(clas)


            reconstructed_model = load_model(choice+".h5")
            out = reconstructed_model.predict(dataset)
            st.write(classes[np.argmax(out)])


    

    else:
        st.subheader("About")
        st.info("Built with Streamlit")
        st.info("by S1DR4G")
        st.text("Siddyant Das")



if __name__ == '__main__':
	main()