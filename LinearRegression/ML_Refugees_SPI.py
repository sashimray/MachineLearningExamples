import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing

filepath= ""
filename= "SPI_Dataset.csv"

learn_rate = 0.001
iters = 9000

def main(data_set):
    
## plot with original values     
    X = data_set['Corruption']
    Y = data_set['Refugees']
    data_set.plot(kind='scatter', x='Corruption', y='Refugees', figsize=(12,8))

## preprocessing stage ###    
    scaled_df = preprocessing_stage(data_set)
    data = pd.DataFrame(scaled_df, columns=['Refugees', 'Corruption'])
#    data = data_set
    print(data)
## plot with new scaled values
    X = data['Corruption']
    Y = data['Refugees']
    
    data.plot(kind='scatter', x='Corruption', y='Refugees', figsize=(12,8))
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]  
    X = data.iloc[:,0:cols-1]  
    y = data.iloc[:,cols-1:cols] 
    
    X = np.matrix(X.values)  
    y = np.matrix(y.values)  
    
    theta = np.matrix(np.array([0,0]))  
    computeCost(X, y, theta) 

    max_theta, cost_hist =gradientDescent(X, y, theta, learn_rate, iters)

    x = np.linspace(data.Corruption.min(), data.Corruption.max(), 100)  
    f = max_theta[0, 0] + (max_theta[0, 1] * x)
    
    fig, ax = plt.subplots(figsize=(12,8))  
    ax.plot(x, f, 'r', label='Prediction')  
    ax.scatter(data.Corruption, data.Refugees, label='Traning Data')  
    ax.legend(loc=2)  
    ax.set_xlabel('Corruption')  
    ax.set_ylabel('Refugees')  
    ax.set_title('Predicted Profit vs. Population Size')  

    fig, ax = plt.subplots(figsize=(12,8))  
    ax.plot(np.arange(iters), cost_hist, 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('Cost')  
    ax.set_title('Error vs. Training Epoch') 
    plt.show()

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

#        print("Iteration:" + str(i) +" "+ str(theta))
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

def preprocessing_stage(my_data):
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(my_data)
    return scaled_data

if __name__== "__main__":
    data_set = pd.read_csv(filepath+filename,header=None,names=['Refugees' , 'Corruption'])
#    data = loadtxt('SPI_corruption.txt', delimiter='\t')
    main(data_set);
