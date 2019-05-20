#import matplotlib
#matplotlib.use('GTKAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, cross_validation as cv

filepath= "C:\\UALR CompSC Masters\\Fall 2018\\Machine Learning\\Hw\\LinearRegression\\"
filename= "dataset1.csv"

learn_rate = 0.01
iterations = 1000

def main(dataset):

    X = dataset['Population']
    Y = dataset['Profit']
    
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,1].values  

    X_train, X_test, Y_train, Y_test = cv.train_test_split(X,Y,test_size=1/3,random_state=0)
    
    m0,b0 = coefficients(X,Y)  ##mean of x_set, y_set
    y0 = target (X, m0,b0)
        
    plt.figure(1)
    plt.title('Predicted profit vs.Population')
    plt.xlabel('Population')
    plt.ylabel('Profit')
    # Plot outputs
    plt.scatter(X_train, Y_train,  color='blue', label='Training Data' )
    plt.plot(X,y0, color='red',linewidth=1, label='Prediction line' )
    plt.legend();
    plt.show();
    
    ## after training, plot the cost function vs iters
    grad, bias, cost_hist = train(X_train, Y_train, m0, b0, learn_rate, iterations)
    plt.figure(2)
    plt.title('Cost Function vs. Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.plot( cost_hist, color='blue',linewidth=1, label='Cost function' )

def coefficients(X,Y):
    # Mean X and Y
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    # Using the formula to calculate gradient and bias
    numer = 0
    denom = 0
    for i in range(len(X)):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2
    m = numer / denom
    b = mean_y - (m * mean_x)
    return (m,b)

def random_coeff(x1, x2, y1, y2):
    m = (x2-x1) / (y2-y1)
    b = y1 - (m * x1)
    return (m,b)

def target (data_X, m0,b0):
    y_plots =[]
    len_X = len(data_X)
    for i in range(len_X):
        y = m0*data_X[i]+b0
        y_plots.append(y)
    
    return y_plots
    
def train(X_set, Y_set, M, B, learning_rate, iters):
    ## minimize the cost funtion 
    cost_history = []

    for i in range(iters):
        M,B = update_gradient(X_set, Y_set, M, B, learning_rate)

        #Calculate cost for updated gradient
        cost = cost_function(X_set, Y_set,  M,B)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print( "iter: "+str(i) + " cost: "+str(cost) + " gradient: " + str(M) + " bias: " + str(B) )

    return M,B, cost_history


def cost_function(X, Y, m, b):  ## y=mx+c
    len_X = len(X)
    total_error = 0.0
    for i in range(len_X):
        total_error += (Y[i] - (m*X[i] + b))**2
    return total_error / len_X

def computeCost(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X)


def update_gradient(data_X, predict_Y, gradient, c, learn_rate):
    gradient_deriv = 0
    c_deriv = 0
    len_X = len(data_X)

    for i in range(len_X):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        gradient_deriv += -2*data_X[i] * (predict_Y[i] - (gradient*data_X[i] + c))
        # -2(y - (mx + b))
        c_deriv += -2*(predict_Y[i] - (gradient*data_X[i] + c))

    # We subtract because the derivatives point in direction of steepest ascent
    gradient -= (gradient_deriv / len_X) * learn_rate
    c -= (c_deriv / len_X) * learn_rate

    return gradient, c

if __name__== "__main__":
    data_set = pd.read_csv(filepath+filename)
    main(data_set);