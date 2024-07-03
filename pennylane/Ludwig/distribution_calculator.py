from scipy.special import rel_entr
import numpy as np
import plotting as plt

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
#https://machinelearningmastery.com/divergence-between-probability-distributions/ to test


def calculate_kl_divergence_log2(p, q):
    return np.sum(p * np.log2(p / q))

def calculate_kl_divergence_log(p, q):
    return np.sum(p * np.log(p / q))
def calculate_kl_divergence_manually(p, q):
    divergence=0
    # Calculate KL divergence
    for i in range(len(p)):
        #print(f"p: {p[i]}")
        #print(f"q: {q[i]}")
        calc=p[i] * np.log2(p[i] / q[i])
        #print(f"calc: {calc}")
        divergence+=calc
        #print(f"div: {divergence}")
    return divergence



def calculate_distribution_with_KLD(predictions,datasets,stepsize, start, end):
    #predictions=datasets
    flat_predictions = np.array(predictions).flatten()
    flat_datasets = np.array(datasets).flatten()
    maximum = max(max(flat_predictions), max(flat_datasets))
    minimum = min(min(flat_predictions), min(flat_datasets))
    #print("MAXIMUM:"+str(maximum))
    #print("MINIMUM:"+str(minimum))
    num_predictions=len(predictions)
    num_inputs=len(datasets)
    lenght_x_prediction=len(predictions[0])
    lenght_x_input=len(datasets[0])
    num_bins = 11          #number of bins the data is divided into
    # Define the bin edges
    bin_edges = np.linspace(minimum-1, maximum+1, num_bins)
    counts_prediction=[]        #initialize the counts array
    counts_input=[]             #initialize the counts array
    counts_input_manuell=[]
    for i in range(lenght_x_prediction):
        data=[]
        for prediction in predictions:
            data.append(prediction[i])
        count=count_values(data,minimum,maximum,num_bins)
        count=count/np.sum(count)       #normalizes the count
        counts_prediction.append(count)
    for i in range(lenght_x_input):
        data=[]
        for dataset in datasets:
            data.append(dataset[i])
        count=count_values(data,minimum,maximum,num_bins)
        count=count/np.sum(count)       #normalizes the count
        counts_input.append(count)
    kl_divergence_bits_list=[]
    kl_divergence_dits_list=[]
    js_divergence_list=[]
    js_distance_list=[]
    for i in range(lenght_x_input):
        bits,dits=kl_divergence(counts_input[i],counts_prediction[i])
        kl_divergence_bits_list.append(bits)
        kl_divergence_dits_list.append(dits)
        js_distance_list.append(jensen_shannon_distance(counts_input[i],counts_prediction[i]))
        js_divergence_list.append(jensen_shannon_divergence(counts_input[i],counts_prediction[i]))
    plt.plot_kl_divergence(value_list=kl_divergence_bits_list, x_start=start, step_size=stepsize, y_label="Kuback-Leibler-divergence in bits", color="orange")
    plt.plot_kl_divergence(kl_divergence_dits_list, start, stepsize, "Kuback-Leibler-divergence in dits", color="red")
    plt.plot_kl_divergence(js_distance_list, start, stepsize, "Jensen-Shannon distance", color="yellow")
    plt.plot_kl_divergence(js_divergence_list, start, stepsize, "Jensen-Shannon divergence", color="green")

def count_values(value_list, minimum_value, maximum_value, bins):
    value_range = maximum_value - minimum_value
    step_size = value_range / bins
    count_list = [0] * bins

    for value in value_list:
        index = int((value - minimum_value) // step_size)
        # Ensure the maximum value is counted in the last bin
        index = min(index, bins-1)
        count_list[index] += 1
    return count_list