from sklearn.datasets import load_digits
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import MinMaxScaler

digits = load_digits()
X,y = digits.data, digits.target
# keep 30% random examples for test
X_train, X_test, y_train, y_test = train_test_split(X,
y, test_size=0.3, random_state=101)

# we scale the data in the range [-1,1]
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
svm = SVC(kernel='linear', C=1)
cv_performance = cross_val_score(svm, X_train, y_train,cv=5)
test_performance = svm.fit(X_train, y_train).score(X_test,y_test)
print ('Cross-validation Support Vector Machine accuracy score: %0.3f,'' test accuracy score: %0.3f'% (np.mean(cv_performance),test_performance))
print (' Standard deviation: +/-'+ str(cv_performance.std() * 2 ) )
