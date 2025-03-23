from pyod.models.iforest import IForest

# anomaly detection with Isolation Forest
clf = IForest()
clf.fit(X)

# prediction
y_pred = clf.predict(X)

# filter anomalies
anomalies = X[y_pred == 1]

# print anomalies and the count of anomalies
print("Anomalies detected: \n", anomalies)

print("\nNumber of anomalies detected: \n", len(anomalies))