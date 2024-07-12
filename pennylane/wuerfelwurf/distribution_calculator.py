import numpy as np
def calculate_kl_divergence_log(p, q):
    return np.sum(p * np.log(p / q))

def calculate_kl_divergence_log2(p, q):
    return np.sum(p * np.log2(p / q))

def kl_divergence(p, q):
    round_precision=6             #rounding to x decimal places

    # Ensures the inputs are numpy arrays
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Add a small constant to avoid division by zero
    p = p + 1e-10
    q = q + 1e-10

    log = round(calculate_kl_divergence_log(p,q),round_precision)     #calculate the KL divergence with log base e in dits(decimal digit)
    log2= round(calculate_kl_divergence_log2(p,q),round_precision)    #calculate the KL divergence with log base 2 in bits(binary digit)

    return log, log2

def jensen_shannon_divergence(p, q):
    """
    Method to calculate the Jensen-Shannon Divergence between two probability distributions
    """
    # First, we need to calculate the M distribution
    m = 0.5 * (p + q)

    # Then, we calculate the KL divergence between p and m, and q and m
    _,kl_pm = kl_divergence(p, m)           #onlyb log2 needed
    _,kl_qm = kl_divergence(q, m)           #only log2 needed

    # Finally, we calculate the Jensen-Shannon Divergence
    js_divergence = 0.5 * (kl_pm + kl_qm)

    return js_divergence

def jensen_shannon_distance(p, q):
    """
    Method to calculate the Jensen-Shannon Distance between two probability distributions
    """
    # Calculate the Jensen-Shannon Divergence
    js_divergence = jensen_shannon_divergence(p, q)

    # The Jensen-Shannon Distance is the square root of the Jensen-Shannon Divergence
    js_distance = np.sqrt(js_divergence)

    return js_distance