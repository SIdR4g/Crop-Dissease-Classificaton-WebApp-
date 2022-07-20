import streamlit as st
import streamlit.components.v1 as stc

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.style as style
import time
from PIL import Image 
from keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle

tfds = tf.data.Dataset

import matplotlib.style as style
style.use('seaborn-darkgrid')

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 

plt.rcParams['font.family'] = "serif"

def main():
    st.title("      Crop dissease classification") #Main Title of WebApp
    st.sidebar.header("Crop Species") # Sidebar consisting of names of species of crops

    menu = ["Home","Apple", "Cherry", "Corn", "Grape", "Peach", "Potato", "Strawberry"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.image('./index.jpeg', width =  720)
        st.subheader("Choose the Crop for which you want to identify the dissease")
    
    elif choice in menu[1:]:

        st.subheader(choice)
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:

            img = load_image(image_file)
            st.image(img,width=480)
            st.write(image_file)

            IMG_SIZE = (1,128, 128,3)
            img_array = np.resize(np.array(img),IMG_SIZE)
            st.write(img_array.shape)

            dataset = tf.data.Dataset.from_tensor_slices(img_array)
            dataset = dataset.batch(1)
            with open(choice+'_class.pkl','rb') as clas:
                classes = pickle.load(clas)

            with st.spinner("Loading....."):
                time.sleep(1)
            reconstructed_model = load_model(choice+".h5")
            out = reconstructed_model.predict(dataset)
            
#             fig = plt.figure(figsize=(10, 4))  
#             sb.set_context('poster')          
#             b = plt.barh( y=classes, width = np.reshape(out, (-1,)), color = 'darkred')
#             b[np.argmax(out)].set_color('green')
#             plt.xlim(0,1)
#             plt.xlabel("Probability ofeach dissease")            
#             st.pyplot(fig)
		
            st.text("The dissease is "+str(classes[np.argmax(out)])+ " with a probability of "+ str(np.max(out)))


    else:
        st.subheader("About")
        st.info("by S1DR4G")
        st.text("Siddyant Das")



if __name__ == '__main__':
	main()
