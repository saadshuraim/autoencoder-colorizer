import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:", tf.config.list_physical_devices('GPU'))
try:
    # Try to load dynamic libraries explicitly to check for errors
    tf.config.experimental.get_memory_info('GPU:0')
except Exception as e:
    print("GPU Error:", e)
