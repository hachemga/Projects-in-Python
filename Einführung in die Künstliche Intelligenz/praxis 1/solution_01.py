import numpy as np


def maximize_mean(mu_vector, sigma_vector):
    #1.1.1
    
    return np.argmax(mu_vector)


def maximize_ucb(mu_vector, sigma_vector):
    #1.1.2

    UCB_vector = mu_vector + 2 * sigma_vector
    return np.argmax(UCB_vector)


def maximize_utility(mu_vector, sigma_vector, utility_function):
   # 1.1.3
    
    expected_utilities = []
    n = 1000


    n_samples_ai = n // mu_vector.size
    

    for i in range(mu_vector.size):
        samples = np.random.normal(mu_vector[i], sigma_vector[i], n_samples_ai)
        utility_values = utility_function(samples)
        apx_expectation =  sum(utility_values) / n 
        expected_utilities.append(apx_expectation)



    return np.argmax(expected_utilities)


def maximize_mean_using_simulator(simulator, A, n):
   # 1.2
    
    
    mu_vector = np.zeros(A)
    UCB_vector = np.zeros(A)
    outcomes = np.zeros((A, n))
    
    # number of plays for every ai
    np_ai = np.zeros(A)

    # play every machine 1 time.
    for i in range(A):
        outcomes[i][0] = simulator(i)
        mu_vector[i] = outcomes[i][0]
        np_ai[i] += 1
    for i in range(A) :
        UCB_vector = mu_vector[i] + np.sqrt((2 * np.log(np.sum(np_ai))) / np_ai[i]) 

    # begin process.
    n = n-A

    while n > 0:
        max_machine = np.argmax(UCB_vector)
        outcomes[max_machine][int(np_ai[max_machine])]= simulator(max_machine)   
        np_ai[max_machine] += 1


        mu_vector[max_machine] = np.sum(outcomes[max_machine]) / np_ai[max_machine]
       
        # UCBs Calc. 
        for i in range(A) :
           UCB_vector = mu_vector[i] + np.sqrt((2 * np.log(np.sum(np_ai))) / np_ai[i]) 

        # Decrease budget
        n = n-1

    return np.argmax(mu_vector)
