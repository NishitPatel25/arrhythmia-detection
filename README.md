# Arrhythmia Detection

This project focuses on detecting arrhythmias using the MIT-BIH ECG dataset. The model leverages both CNN (Convolutional Neural Networks) and LSTM (Long Short-Term Memory) networks for accurate classification of heartbeats, identifying potential arrhythmias.

## Features

- **Arrhythmia Detection:** Detects different types of arrhythmias based on ECG data.
- **CNN + LSTM Model:** Combines the strengths of CNN for feature extraction and LSTM for temporal pattern recognition.
- **Visualization:** Displays ECG signal plots and model performance metrics using Matplotlib.
  
## Technology Stack

- **MIT-BIH ECG Dataset:** Standard dataset for arrhythmia detection.
- **WFDB:** For reading and processing ECG signals from the dataset.
- **CNN + LSTM Model:** A hybrid model combining CNN for spatial feature extraction and LSTM for sequential data modeling.
- **TensorFlow + Keras:** Backend framework for building and training the model.
- **NumPy and Pandas:** For data manipulation and preparation.
- **Matplotlib:** For visualizing ECG signals and model performance.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- WFDB (`pip install wfdb`)
- NumPy (`pip install numpy`)
- Pandas (`pip install pandas`)
- Matplotlib (`pip install matplotlib`)
- TensorFlow and Keras (`pip install tensorflow`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/NishitPatel25/arrhythmia-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd arrhythmia-detection
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1. Download the MIT-BIH ECG dataset from the official source or use the WFDB library to fetch the data.
2. Preprocess the ECG signals (normalizing, filtering, etc.).
3. Train the CNN + LSTM model using the dataset. Modify training parameters in the `train_model.py` file.
4. Once trained, evaluate the model and test it on new ECG signals:
    ```bash
    python test_model.py
    ```

## Dataset

The [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/content/mitdb/) contains annotated ECG recordings from multiple patients. This dataset will be used to train and evaluate the arrhythmia detection model.

## Usage

- Preprocess the raw ECG data using the `data_preprocessing.py` script.
- Train the hybrid CNN + LSTM model to classify heartbeats into normal or abnormal categories.
- Use the model to detect arrhythmias in real-time or batch ECG data.

## Model Overview

The model architecture includes:
- **CNN Layers:** To extract spatial features from the ECG signal.
- **LSTM Layers:** To capture temporal dependencies in the signal for more accurate classification of arrhythmias.
- **Dense Layers:** For final classification of the ECG signals.

## Visualization

Use Matplotlib to visualize:
- ECG signals over time.
- Model accuracy and loss during training.
- Confusion matrix and other evaluation metrics.

## Future Work

- Expand the model to detect more arrhythmia types with greater precision.
- Integrate real-time ECG signal monitoring and arrhythmia detection.
- Explore additional models (e.g., GRU, Transformer-based models) for improved performance.

