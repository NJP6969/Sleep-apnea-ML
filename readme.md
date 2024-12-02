# Apnea CNN Classification Project

This project uses a Convolutional Neural Network (CNN) to classify apnea events in PSG (polysomnography) data. The data is loaded from a preprocessed pickle file (ApneaData.pkl) and split into training and testing sets. The CNN model processes 1D time-series data and outputs a binary classification (0: Non-Apnea, 1: Apnea).

The code evaluates the model using metrics like accuracy, Cohen's kappa, and a classification report, saving predictions in a CSV file.

## Requirements

### Dependencies
Ensure the following Python libraries are installed:

* numpy
* tensorflow
* scikit-learn
* pickle
* csv

Install them using:
```bash
pip install numpy tensorflow scikit-learn
```

## Dataset

The dataset file `ApneaData.pkl` should be placed in the same directory as the script. This file contains preprocessed PSG data where:

* Each sample is an array of length 6001
* The first 6000 elements are features, and the last element is the label (0 or 1)

## How to Run

### Prepare the environment

1. Ensure Python 3.6+ is installed
2. Place the script and ApneaData.pkl in the same folder

### Run the script
```bash
python apnea_cnn.py
```

The script will:
* Load and preprocess the data
* Train the CNN model
* Evaluate performance on the test set
* Save predictions to predictions.csv

### Output

The console will display preprocessing time, training time, accuracy, Cohen's kappa, and a detailed classification report.

The file `predictions.csv` will contain:
```
Index, True Label, Predicted Label
0, 0, 0
1, 1, 1
...
```

## Parameters

### Data Parameters
* filename: (Default: "ApneaData.pkl") Path to the dataset file
* testPercent: (Default: 0.2) Percentage of data used for testing (e.g., 0.2 for 20%)

### Model Parameters
* Input Shape: The model expects inputs of shape (6000, 1), i.e., 6000 time steps with 1 feature

### Training Parameters
* batch_size: (Default: 128) Number of samples per training batch
* epochs: (Default: 60) Number of full passes over the training data
* Optimizer: Adam optimizer with default settings
* Loss Function: Binary Crossentropy for binary classification

## Modifications

### CNN Architecture
Modify the number of convolutional layers, filters, kernel sizes, or dense layers:
```python
cnn.add(layers.Conv1D(128, 3, activation='relu'))
```

Add or adjust pooling layers:
```python
cnn.add(layers.MaxPooling1D(2))
```

### Training Parameters

#### Batch Size
Update batch_size in:
```python
cnn.fit(train_features, train_classes, epochs=epochs, batch_size=128, ...)
```

#### Epochs
Change the number of epochs:
```python
epochs = 60
```

## Performance Metrics

The script calculates:
* Accuracy: Percentage of correctly classified samples
* Cohen's Kappa: Measures agreement between predictions and true labels
* Classification Report: Precision, recall, and F1-score for each class

## Results

After training and testing, results are saved to predictions.csv, and evaluation metrics are printed in the console. Fine-tuning parameters like layers, batch size, or training epochs may improve performance.
