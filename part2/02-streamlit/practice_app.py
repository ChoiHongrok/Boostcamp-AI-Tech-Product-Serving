import streamlit as st
import yaml
import io
from PIL import Image
from predict import load_model, get_prediction
from confirm_button_hack import cache_on_button_press
st.set_page_config(layout='wide')

# st.write('Hello World!')

def main():
    st.title('Mask Classification Model')

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = load_model()
    model.eval()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", 'png'])

    if uploaded_file:
        # TODO: 이미지 view
        # TODO: 예측
        # TODO: 예측 결과 출력
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption='Uploaded Image')
        st.write('Clssifying...')
        _, y_hat = get_prediction(model, image_bytes)
        label = config['classes'][y_hat.item()]

        st.write(f'Prediction Response is {label}')

root_password= 'password'



@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    st.write(type(password))
    return password == root_password

password = st.text_input('password', type='password')

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error("The password is invalid")
# TODO: Streamlit APP 만들기
# TODO: 암호 입력
