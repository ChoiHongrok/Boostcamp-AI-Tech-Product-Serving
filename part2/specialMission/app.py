import streamlit as st
from inference import load_model, predict
from confirm_button_hack import cache_on_button_press
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

def main():
    category_dict = {0:'Background', 1:'General trash', 2:'Paper', 3:'Paper pack', 4:'Metal', 5:'Glass', 
                  6:'Plastic', 7:'Styrofoam', 8:'Plastic bag', 9:'Battery',10:'Clothing'}
    cls2color = {'Background': [0,0,0],
                'General trash': [192,0,128],
                'Paper': [0,128,192],
                'Paper pack': [0,128,64],
                'Metal': [128,0,0],
                'Glass': [64,0,128],
                'Plastic': [64,0,192],
                'Styrofoam': [192,128,64],
                'Plastic bag': [192,192,128],
                'Battery': [64,64,128],
                'Clothing': [128,0,192]}

    MODEL = 'BaseModel'
    MODEL_PATH = '/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/BaseModel_Rotate90/epoch0050_mIoU05815.pth'

    st.title('Trash Segmentation Model')

    model = load_model(MODEL, MODEL_PATH)

    uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        print(uploaded_file)
            
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        st.image(image, caption = 'Uploaded Image')
        st.write('Processing Segmentation....')
        oms = predict(model, image)
        color_oms = np.zeros((512, 512, 3))
        for i in range(11):
            color_oms[oms == i] = cls2color[category_dict[i]]
        color_oms /= 255

        fig = plt.figure(figsize=(17, 14))
        plt.axis('off')
        plt.imshow(color_oms)
        legend = [ patches.Patch(color=[c/255 for c in cls2color[category_dict[i]]], label=f"{i}:{category_dict[i]}" ) for i in range(11) ]
        plt.legend(handles=legend, bbox_to_anchor=(1., 1), loc=2, borderaxespad=0., fontsize=15)
        plt.tight_layout()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        rgba = np.asarray(canvas.buffer_rgba())[:, :, :-1]
        print('rgba shape:', rgba.shape)
        st.image(rgba, caption='segmentation map')


root_password = 'password'

@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    st.write(type(password))
    return password == root_password

password = st.text_input('password', type='password')

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid')
