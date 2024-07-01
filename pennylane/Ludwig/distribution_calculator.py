from scipy.special import rel_entr
import numpy as np

def average_kl_divergence(distributions, smoothing_constant=1e-10):
    # Add the smoothing constant to the distributions
    distributions = [dist + smoothing_constant for dist in distributions]
    n = len(distributions)
    total_kl_div = 0
    print(f"distribution: {distributions}")
    # Calculate KL divergence for each pair of distributions
    for i in range(n):
        for j in range(i+1, n):
            print(f"distr: {distributions[i]}")
            total_kl_div += kl_div(distributions[i], distributions[j]).sum()
            print(f"KL_div: {total_kl_div}")
    # Return the average KL divergence
    print(total_kl_div/(n*(n-1)/2))
    return total_kl_div / (n * (n-1) / 2)

def information_radius(distributions):
    n = len(distributions)
    total_kl_div = 0

    # Calculate KL divergence for each pair of distributions
    for i in range(n):
        for j in range(i+1, n):
            total_kl_div += kl_div(distributions[i], distributions[j]).sum()

    # Return the Information Radius
    return total_kl_div / n

def dissimilarity(distributions):
    n = len(distributions)
    total_dissimilarity = 0

    # Calculate dissimilarity for each pair of distributions
    for i in range(n):
        for j in range(i+1, n):
            total_dissimilarity += np.sqrt(np.sum((distributions[i] - distributions[j])**2))

    # Return the average dissimilarity
    return total_dissimilarity / (n * (n-1) / 2)

def kl_div(p, q):

    # Ensure the inputs are numpy arrays
    p = np.array(p).astype(float)
    q = np.array(q).astype(float)
    #print(type(p))
    #print(p)
    #print(q)
    # Check if inputs are non-negative
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Inputs must be non-negative")

    # Check if inputs have the same shape
    if p.shape != q.shape:
        raise ValueError("Inputs must have the same shape")

    return rel_entr(p, q)