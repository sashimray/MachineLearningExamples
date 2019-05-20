import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing

filepath= "C:\\UALR CompSC Masters\\Fall 2018\\Machine Learning\\Hw\\LinearRegression\\"
filename= "test_LR_refugee.txt"

learn_rate = 0.001
iters = 2000

def main(data_set):
    
## plot with original values     
    X = data_set['Population']
    Y = data_set['Profit']
    data_set.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

## preprocessing stage ###    
    scaled_df = preprocessing_stage(data_set)
    data = pd.DataFrame(scaled_df, columns=['Population', 'Profit'])


## plot with new scaled values
    X = data['Population']
    Y = data['Profit']
    
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]  
    X = data.iloc[:,0:cols-1]  
    y = data.iloc[:,cols-1:cols] 
    
    X = np.matrix(X.values)  
    y = np.matrix(y.values)  
    
    theta = np.matrix(np.array([0,0]))  
    computeCost(X, y, theta) 

    max_theta, cost_hist =gradientDescent(X, y, theta, learn_rate, iters)

    x = np.linspace(data.Population.min(), data.Population.max(), 100)  
    f = max_theta[0, 0] + (max_theta[0, 1] * x)
    
    fig, ax = plt.subplots(figsize=(12,8))  
    ax.plot(x, f, 'r', label='Prediction')  
    ax.scatter(data.Population, data.Profit, label='Traning Data')  
    ax.legend(loc=2)  
    ax.set_xlabel('Population')  
    ax.set_ylabel('Profit')  
    ax.set_title('Predicted Profit vs. Population Size')  

    fig, ax = plt.subplots(figsize=(12,8))  
    ax.plot(np.arange(iters), cost_hist, 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('Cost')  
    ax.set_title('Error vs. Training Epoch') 

def computeCost(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, learn_rate, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((learn_rate / len(X)) * np.sum(term))

        print("Iteration:" + str(i) +" "+ str(theta))
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

def preprocessing_stage(my_data):
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(my_data)
    return scaled_df

if __name__== "__main__":
    data_set = pd.read_csv(filepath+filename,header=None,names=['Population' , 'Profit'])
#    data = loadtxt('SPI_corruption.txt', delimiter='\t')
    main(data_set);
