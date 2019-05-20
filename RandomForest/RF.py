import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

def RF_classification():
    iris = datasets.load_iris()
    
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # sklearn provides the iris species as integer values since this is required for classification
    # here we're just adding a column with the species names to the dataframe for visualisation
    df['species'] = np.array([iris.target_names[i] for i in iris.target])
    sns.pairplot(df, hue='species')
    
    
    X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target, test_size=0.5, stratify=iris.target, random_state=123456)
    
    
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
    rf.fit(X_train, y_train)
    
    
    predicted = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
    print(f'Mean accuracy score: {accuracy:.3}')
    # *** OUTPUT ****
    # Out-of-bag score estimate: 0.973
    # Mean accuracy score: 0.933
    
    cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names, index=iris.target_names)
    sns.heatmap(cm, annot=True)

def RF_regression():
    boston = datasets.load_boston()
    features = pd.DataFrame(boston.data, columns=boston.feature_names)
    targets = boston.target
    
    
    X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)
    
    pca = PCA()
    pca.fit(X_train)
    cpts = pd.DataFrame(pca.transform(X_train))
    x_axis = np.arange(1, pca.n_components_+1)
    
    pca_scaled = PCA()
    pca_scaled.fit(X_train_scaled)
    cpts_scaled = pd.DataFrame(pca.transform(X_train_scaled))
    
    rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
    rf.fit(X_train, y_train)
    
    
    predicted_train = rf.predict(X_train)
    predicted_test = rf.predict(X_test)
    test_score = r2_score(y_test, predicted_test)
    spearman = spearmanr(y_test, predicted_test)
    pearson = pearsonr(y_test, predicted_test)
    print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    print(f'Test data R-2 score: {test_score:>5.3}')
    print(f'Test data Spearman correlation: {spearman[0]:.3}')
    print(f'Test data Pearson correlation: {pearson[0]:.3}')
    
    # *** OUTPUT ****
    #Out-of-bag R-2 score estimate: 0.841
    #Test data R-2 score: 0.886
    #Test data Spearman correlation: 0.904
    #Test data Pearson correlation: 0.942

if __name__== "__main__":
    RF_classification()
    RF_regression()