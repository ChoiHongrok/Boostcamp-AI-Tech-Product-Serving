from importlib import import_module

import torch
import numpy as np
from glob import glob
import streamlit as st

category_dict = {0:'Background', 1:'General trash', 2:'Paper', 3:'Paper pack', 4:'Metal', 5:'Glass', 
                  6:'Plastic', 7:'Styrofoam', 8:'Plastic bag', 9:'Battery',10:'Clothing'}

@st.cache
def load_model(MODEL, model_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_cls = getattr(import_module("model"), MODEL)
    model = model_cls().to(device)
    
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)
    print('====================================================================================================================')
    print('====================================================================================================================')
    print(f'load checkpoint from {model_path}')
    print('====================================================================================================================')
    print('====================================================================================================================')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model

def predict(model, img):

    img = img.resize((512, 512))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Start prediction.')
    
    model.eval()
    
    with torch.no_grad():
        tensor = torch.Tensor(np.array(img)).to(device).unsqueeze(0).permute(0, 3, 1, 2).clip(0, 255)/255
        print('input_shape: ', tensor.shape)
        out = model(tensor)['out']
        print('output_shape: ', out.shape)
        oms = torch.argmax(out, dim=1).squeeze(0).detach().cpu().numpy()
        print('oms_shape: ', oms.shape)
                    
    return oms