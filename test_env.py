import cv2
import tensorflow as tf
import sys
print("Python:", sys.executable)
try:
    print("CV2:", cv2.__version__)
except Exception as e:
    print("CV2 Error:", e)

try:
    print("TF:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("TF Error:", e)
