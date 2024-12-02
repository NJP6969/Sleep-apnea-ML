import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
import time
import pickle
import csv

# --------------------------- LOAD DATA ---------------------------

filename = "ApneaData.pkl"  # The data is stored in a pickle file (binary format for Python objects).
testPercent = 0.2  # Percentage of the dataset to be used for testing.
features = []  # Placeholder for input features.
classes = []  # Placeholder for target labels (classes).

# Load the data from the pickle file.
t = time.time()  # Start a timer to measure preprocessing time.
with open(filename, 'rb') as f:  # Open the pickle file in binary read mode.
    data = pickle.load(f)  # Load the data into a Python object (list of arrays).
np.random.shuffle(data)  # Shuffle the dataset to ensure randomness in training and testing splits.

# Split the data into features (input data) and classes (output labels).
for row in data:
    features.append(row[:-1])  # All values except the last one are features.
    classes.append(row[-1])  # The last value is the class label.

# Calculate dataset size and split it into training and testing sets.
inputLength = len(features)  # Total number of samples.
testLength = int(inputLength * testPercent)  # Number of test samples (20% of total data).
train_features, train_classes = features[:-testLength], classes[:-testLength]  # Training set.
test_features, test_classes = features[-testLength:], classes[-testLength:]  # Testing set.

print("Data Preprocessing Time: ", (time.time() - t))  # Print the time taken to preprocess the data.

# --------------------------- DATA PREPARATION ---------------------------

# Convert the lists of features and classes into numpy arrays for compatibility with TensorFlow.
train_features = np.array(train_features)
test_features = np.array(test_features)
train_classes = np.array(train_classes)
test_classes = np.array(test_classes)

# Reshape the feature arrays to be compatible with the CNN input format.
# Each sample should have a shape of (6000, 1), where 6000 is the input length, and 1 is the single channel.
train_features = train_features.reshape(-1, 6000, 1)
test_features = test_features.reshape(-1, 6000, 1)

# --------------------------- CNN MODEL DEFINITION ---------------------------

cnn = models.Sequential()  # Initialize a Sequential model (layers are stacked sequentially).

# Add the first convolutional block.
cnn.add(layers.Conv1D(32, 3, activation='relu', input_shape=(6000, 1)))  # Conv1D with 32 filters and a kernel size of 3.
cnn.add(layers.MaxPooling1D(2))  # Max pooling with a pool size of 2 to downsample the data.

# Add additional convolutional layers for feature extraction.
cnn.add(layers.Conv1D(64, 3, activation='relu'))  # Conv1D with 64 filters.
cnn.add(layers.MaxPooling1D(2))  # Downsample further.

cnn.add(layers.Conv1D(128, 3, activation='relu'))  # Conv1D with 128 filters.
cnn.add(layers.MaxPooling1D(2))  # Further downsample.

# Add a global average pooling layer to reduce dimensions before passing to dense layers.
cnn.add(layers.GlobalAveragePooling1D())

# Add dense layers for classification.
cnn.add(layers.Dense(64, activation='relu'))  # Fully connected layer with 64 neurons.
cnn.add(layers.Dense(1, activation='sigmoid'))  # Output layer with 1 neuron for binary classification.

# Compile the model with the Adam optimizer, binary cross-entropy loss, and accuracy as a metric.
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the model's architecture.
cnn.summary()

# --------------------------- TRAINING ---------------------------

# Define training parameters.
batch_size = 128  # Process 128 samples at a time during training.
epochs = 60  # Train for 60 iterations over the entire dataset.

# Start training and record the time.
start_time = time.time()
cnn.fit(
    train_features,  # Training data.
    train_classes,  # Training labels.
    epochs=epochs,  # Number of epochs.
    batch_size=batch_size,  # Batch size.
    validation_data=(test_features, test_classes)  # Validation data to monitor performance.
)
print("Training Time: ", (time.time() - start_time))  # Print the training time.

# --------------------------- EVALUATION ---------------------------

# Predict the classes of the test dataset.
pred_classes = cnn.predict(test_features)

# Convert predictions into binary values (0 or 1) based on a threshold of 0.5.
pred_classes = (pred_classes > 0.5).astype(int)

# Calculate the accuracy score.
accuracy = accuracy_score(test_classes, pred_classes) * 100  # Accuracy as a percentage.

# Calculate Cohen's kappa score (a measure of agreement).
kappa = cohen_kappa_score(test_classes, pred_classes)

# Print the results.
print("Accuracy: {:.2f}%".format(accuracy))
print("Cohen's Kappa: {:.2f}".format(kappa))

# Print a detailed classification report with precision, recall, and F1-score.
print("Classification Report:\n", classification_report(test_classes, pred_classes))

# --------------------------- SAVE PREDICTIONS ---------------------------

# Save the predictions, along with true labels, to a CSV file for further analysis.
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'True Label', 'Predicted Label'])  # Write header.
    for i in range(len(test_classes)):
        writer.writerow([i, test_classes[i], pred_classes[i][0]])  # Write each sample's data.
