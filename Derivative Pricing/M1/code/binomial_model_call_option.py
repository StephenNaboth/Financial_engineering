import numpy as np


def binomial_call_full(S_ini, K, T, r, u, d, N):
    """
    This function takes 'S_ini', 'K', 'T', 'r', 'u', 'd' and 'N' values and returns call option price at t = 0 'C[0,0]', \
        call option payoff 'C' and underlying price evolution 'S'

    Args:
        S_init : float
                 Initial stock price
        K      : float
                 strike price
        T      : int
                 time horizon
        r      : float
                 risk-free rate
        u      : float
                 upward movement
        d      : float
                 downward movement
        N      : int
                 number of steps in the tree

    Returns:
        C[0, 0] : float
                  call option price at t = 0
        C       : matrix
                  call option payoff
        S       : matrix
                  underlying price evolution
    """
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # Risk neutral probabilities (probs)
    C = np.zeros([N + 1, N + 1])  # Call prices
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        C[N, i] = max(S_ini * (u ** (i)) * (d ** (N - i)) - K, 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            # call option value at each node
            C[j, i] = np.exp(-r * dt) * \
                (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return C[0, 0], C, S


# example
# let the user input values
S_ini = float(input("Enter stock price (say between 100): "))
K = float(input("Enter strike price (say 90): "))
T = int(input("Enter the time horizon (say 3): "))
r = float(input("Enter the risk free rate (say 0): "))
u = float(input("Enter the upward movement value (say 1.2): "))
d = float(input("Enter the downward movement value (say 0.8): "))
N = int(input("Enter the number of stepg (say 3): "))

call_price, C, S = binomial_call_full(S_ini, K, T, r, u, d, N)
print("Underlying Price Evolution:\n", S)
print("Call Option Payoff:\n", C)
print("Call Option Price at t=0: ", "{:.2f}".format(call_price))
