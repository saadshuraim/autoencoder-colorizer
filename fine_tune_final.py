import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. Enable Mixed Precision
mixed_precision.set_global_policy('mixed_float16')

# 2. Paths
MODEL_PATH = r"E:\University\Gen AI\Project\Autoencoder-Colorization\model\AutoEncoder.h5"
DATASET_DIR = r"E:\University\Gen AI\Project\dataset"
GRAY_PATH = os.path.join(DATASET_DIR, "l", "gray_scale.npy")
AB_DIR = os.path.join(DATASET_DIR, "ab", "ab")

# 3. Load Data
print("Loading dataset...")
gray_scale = np.load(GRAY_PATH)
# Ensure L is (N, H, W, 1)
if len(gray_scale.shape) == 3:
    gray_scale = np.expand_dims(gray_scale, axis=-1)

ab_list = []
# User reported only ab1, ab2, ab3 exist
for i in range(1, 4):
    path = os.path.join(AB_DIR, f"ab{i}.npy")
    ab_list.append(np.load(path))
ab_full = np.concatenate(ab_list, axis=0)

# Sync gray_scale length with available ab data
gray_scale = gray_scale[:len(ab_full)]

print(f"Gray shape: {gray_scale.shape}")
print(f"AB shape: {ab_full.shape}")

# 4. Split Data (10% Validation)
VAL_SIZE = int(len(gray_scale) * 0.1)
train_L = gray_scale[:-VAL_SIZE]
train_ab = ab_full[:-VAL_SIZE]
val_L = gray_scale[-VAL_SIZE:]
val_ab = ab_full[-VAL_SIZE:]

# 5. Dataset Pipeline
def preprocess_numpy(L, ab):
    # L: (H,W,1), ab: (H,W,2)
    # Combine to Lab
    img_lab = np.concatenate([L, ab], axis=-1) # (H,W,3)
    
    # Random Crop to 120x120
    H, W, _ = img_lab.shape
    crop_size = 120
    if H > crop_size and W > crop_size:
        start_x = np.random.randint(0, W - crop_size + 1)
        start_y = np.random.randint(0, H - crop_size + 1)
        img_crop = img_lab[start_y:start_y+crop_size, start_x:start_x+crop_size, :]
    else:
        img_crop = cv2.resize(img_lab, (crop_size, crop_size))

    # Prepare Input: L channel repeated 3 times, normalized
    L_crop = img_crop[:, :, 0:1]
    input_img = np.repeat(L_crop, 3, axis=-1)
    input_img = input_img.astype(np.float32) / 255.0

    # Prepare Target: Lab -> RGB, normalized
    # cv2 expects uint8 for correct Lab conversion if data is 0-255
    img_crop_uint8 = img_crop.astype(np.uint8)
    rgb_crop = cv2.cvtColor(img_crop_uint8, cv2.COLOR_Lab2RGB)
    target_img = rgb_crop.astype(np.float32) / 255.0

    return input_img, target_img

def tf_preprocess(L, ab):
    # Wrapper for numpy function
    input_img, target_img = tf.numpy_function(preprocess_numpy, [L, ab], [tf.float32, tf.float32])
    input_img.set_shape([120, 120, 3])
    target_img.set_shape([120, 120, 3])
    return input_img, target_img

def create_dataset(L_data, ab_data, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices((L_data, ab_data))
    ds = ds.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if is_train:
        ds = ds.shuffle(25000)
    ds = ds.batch(64)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_L, train_ab, is_train=True)
val_ds = create_dataset(val_L, val_ab, is_train=False)

# 6. Load and Setup Model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Freeze encoder
model.layers[0].trainable = False

# 7. Compile
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.Huber()
)

# 8. Callbacks
callbacks = [
    ModelCheckpoint("model/colorization_finetuned_best.h5", save_best_only=True, monitor='val_loss', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

# 9. Train
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    verbose=1,
    callbacks=callbacks
)

# 10. Save Final
model.save("model/colorization_finetuned_best.keras")
print("Training complete. Model saved.")

# === BONUS: UNCOMMENT TO ADD PERCEPTUAL LOSS LATER ===
# from tensorflow.keras.applications import InceptionResNetV2
# perc_model = InceptionResNetV2(weights=r"E:\University\Gen AI\Project\dataset\inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5", include_top=False, input_shape=(120,120,3))
# perc_model.trainable = False
# perceptual_loss = lambda y_true, y_pred: tf.reduce_mean(tf.abs(perc_model(y_true) - perc_model(y_pred)))
