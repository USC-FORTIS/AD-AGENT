
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.data_loader import DataLoader
from pyod.models.abod import ABOD
from sklearn.metrics import roc_auc_score, average_precision_score

# Initialize DataLoader
dataloader_train = DataLoader(filepath='./data/glass_train.mat', store_script=True, store_path='train_data_loader.py')
dataloader_test = DataLoader(filepath='./data/glass_test.mat', store_script=True, store_path='test_data_loader.py')

# Load data
X_train, y_train = dataloader_train.load_data(split_data=False)
X_test, y_test = dataloader_test.load_data(split_data=False)

# Initialize ABOD
model = ABOD(contamination=0.2, n_neighbors=10, method='fast')

# Train the model
model.fit(X_train)

# Get training outlier scores
train_scores = model.decision_scores_

# Get test outlier scores
test_scores = model.decision_function(X_test)

# Calculate AUROC and AUPRC
auroc = roc_auc_score(y_test, test_scores)
auprc = average_precision_score(y_test, test_scores)

# Print AUROC and AUPRC
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")

# Record and print failed predictions
predictions = model.predict(X_test)
for i, (pred, true_label) in enumerate(zip(predictions, y_test)):
    if pred != true_label:
        print(f"Failed prediction at point {X_test[i].tolist()} with true label {true_label}")
                