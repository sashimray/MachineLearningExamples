from sklearn.datasets import load_digits
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

digits = load_digits()
X,y = digits.data, digits.target
# Keep 30% random examples for test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

# we scale the data in the range [-1,1]
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=85, alpha=1e-4,
                    solver='sgd',  tol=1e-4, random_state=1,
                    learning_rate_init=.1)
cv_performance = cross_val_score(mlp, X_train, y_train,cv=5)
test_performance = mlp.fit(X_train, y_train).score(X_test,y_test)
print ('Cross-validation Neural Network accuracy score: %0.3f,'' test accuracy score: %0.3f'% (np.mean(cv_performance),test_performance))
print ('Standard deviation: +/-'+ str(cv_performance.std() * 2 ) )
