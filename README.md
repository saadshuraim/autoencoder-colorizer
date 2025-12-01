# GenAI Landscape Colorizer ✨

This project implements an Image Colorization system using Autoencoders. It includes a Streamlit web application for easy interaction and scripts for fine-tuning the model on custom datasets.

## 🚀 Features

- **Web Interface**: A user-friendly Streamlit app to upload grayscale images and view colorized results instantly.
- **Autoencoder Model**: Utilizes a deep learning autoencoder architecture for image colorization.
- **Fine-tuning**: Scripts provided to fine-tune the model on your own dataset (supports `.npy` format).
- **Mixed Precision**: Optimized for performance using TensorFlow's mixed precision policies.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/saadshuraim/autoencoder-colorizer.git
    cd Autoencoder-Colorization
    ```

2.  **Activate the existing virtual environment**:
    A virtual environment named `myenv` is already created for this project. You just need to activate it.
    ```bash
    # Windows
    myenv\Scripts\activate
    # macOS/Linux
    source myenv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage

### Running the Web App

To launch the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to the URL provided (usually `http://localhost:8501`).

### Training / Fine-tuning

The model was trained using a combination of datasets to ensure robust colorization capabilities:

1.  **MIRFLICKR-25k Dataset**:
    -   Utilized pre-processed `.npy` files: `gray_scale.npy` (L channel) and `ab1.npy`, `ab2.npy`, `ab3.npy` (ab channels).
    -   Total images: ~25,000.
2.  **Custom Folder Dataset**:
    -   Additional image pairs from `train_black` and `train_color` folders.
    -   Total images: ~5,000.

**Training Details:**
-   **Architecture**: Improved U-Net style Autoencoder with skip connections.
-   **Optimization**:
    -   **Mixed Precision**: Enabled (`mixed_float16`) for faster training and lower VRAM usage.
    -   **GPU**: Trained on NVIDIA RTX 3070.
    -   **Loss Function**: Mean Squared Error (MSE).
    -   **Optimizer**: Adam.
-   **Preprocessing**:
    -   Images resized/cropped to 128x128 pixels.
    -   L channel (grayscale) used as input (repeated 3 times to match RGB shape).
    -   Lab color space converted to RGB for target output.
-   **Strategy**:
    -   Infinite dataset generators for continuous training.
    -   Callbacks: `ModelCheckpoint` (saves best model), `ReduceLROnPlateau` (adjusts learning rate), `EarlyStopping`.

To fine-tune the model on your dataset:

1.  Ensure your dataset is prepared (grayscale and ab channels in `.npy` format or image folders).
2.  Update the paths in `fine_tune_final.py` or use the logic from `newKoshish.ipynb`:
    ```python
    MODEL_PATH = r"path/to/your/model.h5"
    DATASET_DIR = r"path/to/your/dataset"
    ```
3.  Run the training script:
    ```bash
    python fine_tune_final.py
    ```

## 📂 Project Structure

- `app.py`: Main Streamlit application script.
- `fine_tune_final.py`: Script for fine-tuning the autoencoder model.
- `packages/`: Contains utility modules.
    - `colorizer.py`: Model loading and prediction logic.
    - `normalizer.py`: Image normalization utilities.
- `model/`: Directory to store trained models (`AutoEncoder.h5`).
- `requirements.txt`: List of Python dependencies.

## 📝 Notes

- The model is trained to work with 128x128 pixel images.
- The `FixedConv2DTranspose` class handles specific loading requirements for the Keras model.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
