# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""This function takes actual and predicted ratings and compute total mean square error(mse) in observed ratings.
"""
def computeError(R,predR):
    
    """Your code to calculate MSE goes here"""
    mse=0
    i=len(R)
    j=len(R[0])
    for x in range(i):
        for y in range(j):
            if R[x][y]!=0:
                error=(R[x][y]-predR[x][y])
                mse+=error
    
    return mse
    


"""
This fucntion takes P (m*k) and Q(k*n) matrices alongwith user bias (U) and item bias (I) and returns predicted rating. 
where m = No of Users, n = No of items
"""
def getPredictedRatings(P,Q,U,I):
    m=len(P)
    n=len(Q[0])
    matrix=[]
    for i in range(m):
        row=[]
        for j in range(n):
            dot=P[i,:].dot(Q[:,j])
            predicted=U[i]+I[j]+dot
            row.append(predicted)
        matrix.append(row)
    return matrix
    
    
"""This fucntion runs gradient descent to minimze error in ratings by adjusting P, Q, U and I matrices based on gradients.
   The functions returns a list of (iter,mse) tuple that lists mse in each iteration
"""
def runGradientDescent(R,P,Q,U,I,iterations,alpha):
   
    stats = []
    sample=[]
    for x in range(len(R)):
        for y in range(len(R[0])):
            if R[x][y]!=0:
                tup=(x,y,R[x][y]) #contains row, col and value of R's entry != 0
                sample.append(tup)
    for i in range(iterations):
        for x,y,r in sample:
            predict_matrix=getPredictedRatings(P,Q,U,I)
            prediction=predict_matrix[x][y]
            error=r-prediction

            #updating biases
            U[x]+=alpha*(2*error)
            I[y]+=alpha*(2*error)
            
            #updating P and Q
            oldP=P
            P[x,:]+=alpha*(2*error*Q[:,y])
            Q[:,y]+=alpha*2*error*P[x,:]
        mse=computeError(R,predict_matrix)
        err=(i,mse)
        stats.append(err)
    
    """"finally returns (iter,mse) values in a list"""
    #print(stats)
    return stats
    
""" 
This method applies matrix factorization to predict unobserved values in a rating matrix (R) using gradient descent.
K is number of latent variables and alpha is the learning rate to be used in gradient decent
"""    

def matrixFactorization(R,k,iterations, alpha):

    """Your code to initialize P, Q, U and I matrices goes here. P and Q will be randomly initialized whereas U and I will be initialized as zeros. 
    Be careful about the dimension of these matrices
    """
    users,items=R.shape
    P = np.random.normal(scale=1./k, size=(users, k))
    Q = np.random.normal(scale=1./k, size=(items, k))
    Q=Q.T#Transpose this so that its k*n
    U = np.zeros(users)
    I = np.zeros(items)
    global_bias=np.mean(R[np.where(R != 0)])
    #Run gradient descent to minimize error
    stats = runGradientDescent(R,P,Q,U,I,iterations,alpha)
    
    print('P matrx:')
    print(P)
    print('Q matrix:')
    print(Q)
    print("User bias:")
    print(U)
    print("Item bias:")
    print(I)
    print("P x Q:")
    print(getPredictedRatings(P,Q,U,I))
    plotGraph(stats)
       
    
def plotGraph(stats):
    i = [i for i,e in stats]
    e = [e for i,e in stats]
    plt.plot(i,e)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Square Error")
    plt.show()    
    
""""
User Item rating matrix given ratings of 5 users for 6 items.
Note: If you want, you can change the underlying data structure and can work with starndard python lists instead of np arrays
We may test with different matrices with varying dimensions and number of latent factors. Make sure your code works fine in those cases.
"""
R = np.array([
[5, 3, 0, 1, 4, 5],
[1, 0, 2, 0, 0, 0],
[3, 1, 0, 5, 1, 3],
[2, 0, 0, 0, 2, 0],
[0, 1, 5, 2, 0, 0],
])

k = 3
alpha = 0.01
iterations = 500

matrixFactorization(R,k,iterations, alpha)
