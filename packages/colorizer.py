import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2DTranspose

class FixedConv2DTranspose(Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

AutoEncoder = load_model('./model/AutoEncoder.h5', custom_objects={'Conv2DTranspose': FixedConv2DTranspose}, compile=False)


def predict(arr):
    colorized_img = AutoEncoder.predict(arr)
    return colorized_img