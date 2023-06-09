from dataclasses import replace
from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp
import sys
dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    mu = myTheta.mu
    omega = myTheta.omega
    Sigma = myTheta.Sigma
    d = Sigma.shape[1]
    log_numerator = -0.5 * np.sum(np.square(x - mu[m]) / Sigma[m],axis=1)
    pi_term = (0.5 * d) * np.log(2 * np.pi)
    #  sum of log product from 1 to d 
    sqroot_term = 0.5 * np.sum(np.log(Sigma[m]))
    log_denominator = pi_term + sqroot_term
    result = log_numerator - log_denominator
    
    return result

    
def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''

    mu = myTheta.mu
    omega = myTheta.omega
    Sigma = myTheta.Sigma
    M = omega.shape[0]
    T = x.shape[0]
    # the M th log_b
    log_bmx = log_b_m_x(m,x,myTheta)
    log_numerator = np.log(omega[m]) + log_bmx
    
    # collection of M numbers of logb  
    log_bms = np.zeros((M,T))
    for i in range(M):
        log_bms[i] = log_b_m_x(i,x,myTheta)
    log_arr = np.array(log_bms)
    log_denominator = logsumexp(np.log(omega) + log_arr)
    
    result = log_numerator - log_denominator
    return result 
    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''

    omega = myTheta.omega
    log_omega = np.log(omega)
    log_p = logsumexp(log_omega + log_Bs)

    return np.sum(log_p)

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    T = X.shape[0]
    myTheta = theta( speaker, M, X.shape[1] )
    # Initialize theta
    # random choice
    rand = np.random.choice(T, M, replace = False)
    myTheta.mu = X[rand]

    # initialize identity matrix
    myTheta.Sigma= np.identity(X.shape[1])
    
    # initialize omega evenly
    myTheta.omega = np.full((M,1),1/M)

    # Initialize param
    i = 0
    prev_L = float("-inf")
    improvement = float("inf")

    while i <= maxIter and improvement >= epsilon:
        # compute Intermediate results
        # initalize two np arrays
        eq1_arr = np.zeros((M,T))
        eq2_arr = np.zeros((M,T))
        
        for i in range(M):
            eq1_arr[i] = log_b_m_x(i,X,myTheta)
        for j in range(M):
            eq2_arr[j] = log_p_m_x(j,X,myTheta)

        L = logLik (eq1_arr, myTheta)
        # update parameters for m componenets
       
        p_mx = np.exp(eq2_arr)
        sum_p = np.sum(p_mx)
        myTheta.omega = sum_p / T
        myTheta.mu = np.matmal(p_mx, X) / sum_p
        myTheta.Sigma = np.matmal(p_mx, np.square(X)) / sum_p - np.square(myTheta.mu)
    
        improvement = L - prev_L
        prev_L = L
        i = i + 1

    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    log_likely_set = []
    best_likelyhood = float("-inf")

    T = mfcc.shape[0]
    
    #  compute all log likelyhood in a model
    for i in range(len(models)):
        theta = models[i]
        M = theta.omega.shape[0]
        # log _bs array
        log_bs = np.zeros((M,T))
        for j in range(M):
            log_bs[j] = log_b_m_x(j,mfcc,theta)
        
        ith_likely_hood = logLik(log_bs,theta)
        log_likely_set.append((theta.name, ith_likely_hood))
        if ith_likely_hood > best_likelyhood:
            bestModel = i
            best_likelyhood = ith_likely_hood

    # sorted
    log_likely_set.sort(key = lambda x: x[1], reverse=True)

    print(models[correctID].name)
    for i in range(k):
        print(log_likely_set[i][0], log_likely_set[i][1])

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    # print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    sys.stdout = open('gmmLiks.txt', 'w')
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print(accuracy)
    sys.stdout.close()

