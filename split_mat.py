import scipy.io
from sklearn.model_selection import train_test_split

data = scipy.io.loadmat('./data/glass.mat')

X = data['X']
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scipy.io.savemat('./data/glass_train.mat', {'X': X_train, 'y': y_train})

scipy.io.savemat('./data/glass_test.mat', {'X': X_test, 'y': y_test})
