import csv
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

# Load predictions from the CSV file
true_labels, predicted_labels = [], []
with open('prediction_results.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        true_labels.append(int(row[0]))
        predicted_labels.append(int(row[1]))

# Compute metrics
accuracy = accuracy_score(true_labels, predicted_labels)
kappa = cohen_kappa_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, target_names=["Non-Apnea", "Apnea"])

# Display results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Cohen's Kappa: {kappa:.2f}")
print("\nClassification Report:")
print(report)
