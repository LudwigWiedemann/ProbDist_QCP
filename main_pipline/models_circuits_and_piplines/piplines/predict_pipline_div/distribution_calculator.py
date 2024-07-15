from scipy.special import rel_entr
import numpy as np
import pennylane as qml
from main_pipline.input.div.logger import Logger
import main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.distribution_plotter as plt
from pathlib import Path


num_bins=10 #number of bins the data is divided into
num_bad_histograms=0 #number of bad histograms to be plotted
num_good_histograms=0   #number of good histograms to be plotted


def average_kl_divergence(distributions, smoothing_constant=1e-10):
    '''calculates the kl_divergence of multiple Inputs (unused)'''
    # Add the smoothing constant to the distributions
    distributions = [dist + smoothing_constant for dist in distributions]
    n = len(distributions)
    total_kl_div = 0
    print(f"distribution: {distributions}")
    # Calculate KL divergence for each pair of distributions
    for i in range(n):
        for j in range(i+1, n):
            print(f"distr: {distributions[i]}")
            #total_kl_div += kl_div(distributions[i], distributions[j]).sum()
            print(f"KL_div: {total_kl_div}")
    # Return the average KL divergence
    print(total_kl_div/(n*(n-1)/2))
    return total_kl_div / (n * (n-1) / 2)



def information_radius(distributions):
    '''calculates the information radius for multiple inputs (unused)'''
    n = len(distributions)
    total_kl_div = 0
    # Calculate KL divergence for each pair of distributions
    for i in range(n):
        for j in range(i+1, n):
            #total_kl_div += kl_div(distributions[i], distributions[j]).sum()
            None
    # Return the Information Radius
    return total_kl_div / n

def dissimilarity(distributions):
    '''calculates the dissililarity for multiple inputs (unused)'''
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
    """
    Metod to calculate the Kullback-Leibler-Divergence from q to base p, ensures values working
    :param p: base value
    :param q: value to compare to p
    :return: kl-divergence float
    """
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
'''link: https://machinelearningmastery.com/divergence-between-probability-distributions/ to test'''


def calculate_kl_divergence_log2(p, q):
    """
    Metod to calculate the Kullback-Leibler-Divergence from q to base p in bits
    :param p: base value
    :param q: value to compare to p
    :return: kl-divergence float in bits
    """
    return np.sum(p * np.log2(p / q))

def calculate_kl_divergence_log(p, q):
    """
    Metod to calculate the Kullback-Leibler-Divergence from q to base p in dits
    :param p: base value
    :param q: value to compare to p
    :return: kl-divergence float in dits
    """
    return np.sum(p * np.log(p / q))
def calculate_kl_divergence_manually(p, q):
    """
    Metod to calculate the Kullback-Leibler-Divergence from q to base p in bits (not used, just to ensure result)
    :param p: base value
    :param q: value to compare to p
    :return: kl-divergence float in bits
    """
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



def calculate_distribution_with_KLD(predictions,datasets,stepsize, start, end, logger):
    """
    calculates and plots KLD in bits, dits JSDivergence and JSDistance for given prediction- and dataset-list
    :param predictions: list[lists[float]]: output-distribution
    :param datasets: list[lists[float]]: input-distribution
    :param stepsize: float: stepsize between values in prediction and dataset for plotting
    :param start: float: startvalue for x of plot
    :param end: float: endvalue for x of plot
    :param logger: logger to print to
    :return: None
    """
    flat_predictions = np.array(predictions).flatten()
    flat_datasets = np.array(datasets).flatten()
    #Defines the global extrema of all distributions
    maximum = max(max(flat_predictions), max(flat_datasets))
    minimum = min(min(flat_predictions), min(flat_datasets))

    datasets = [datasets.tolist()]
    lenght_x_prediction=len(predictions[0])
    lenght_x_input=len(datasets[0])

    counts_prediction=[]        #initialize the counts array
    counts_input=[]             #initialize the counts array



    #transform a list of prediction lists into a list of normalized histograms
    counts_prediction=histogram_from_list(lenght_x_prediction, predictions, minimum, maximum, num_bins)
    counts_input=histogram_from_list(lenght_x_input, datasets, minimum, maximum, num_bins)

    #initialize the lists for the KL divergence in bits and dits, the Jensen-Shannon-Distance and the Jensen-Shannon-Divergence
    kl_divergence_bits_list=[]
    kl_divergence_dits_list=[]
    js_divergence_list=[]
    js_distance_list=[]

    #for every x_step calculate the KL divergence in bits and dits, the Jensen-Shannon-Distance and the Jensen-Shannon-Divergence
    for i in range(lenght_x_input):
        bits,dits=kl_divergence(counts_input[i],counts_prediction[i])
        kl_divergence_bits_list.append(bits)
        kl_divergence_dits_list.append(dits)
        js_distance_list.append(jensen_shannon_distance(counts_input[i],counts_prediction[i]))
        js_divergence_list.append(jensen_shannon_divergence(counts_input[i],counts_prediction[i]))
        #plot the distribution at i if the KL divergence is very low or very high if enabled
        c=0
        if bits<2 and c<num_good_histograms:
            count_input_at_i=counts_input[i]
            count_prediction_at_i=counts_prediction[i]
            logger.info(f"count_input_at_i:{i}: {count_input_at_i}")
            logger.info(f"count_prediction_at_i:{i}: {count_prediction_at_i}")
            logger.info(f"KL divergence in bits at i={i}: {bits}")
            plt.plot_divergence_at_i(count_input_at_i, count_prediction_at_i, i, bits,logger)
            c+=1
        c=0
        if bits>15 and c<num_bad_histograms:
            count_input_at_i=counts_input[i]
            count_prediction_at_i=counts_prediction[i]
            logger.info(f"count_input_at_i:{i}: {count_input_at_i}")
            logger.info(f"count_prediction_at_i:{i}: {count_prediction_at_i}")
            logger.info(f"KL divergence in bits at i={i}: {bits}")
            plt.plot_divergence_at_i(count_input_at_i, count_prediction_at_i, i, bits,logger)
            c+=1
    #plot the KL divergence in bits, dits, Jensen-Shannon-Distance and Jensen-Shannon-Divergence
    plt.plot_kl_divergence(value_list=kl_divergence_bits_list, x_start=end, step_size=stepsize, y_label="Kullback-Leibler-Divergence in bits", color="orange",logger=logger)
    plt.plot_kl_divergence(kl_divergence_dits_list, end, stepsize, "Kullback-Leibler-Divergence in dits", color="red",logger=logger)
    plt.plot_kl_divergence(js_distance_list, end, stepsize, "Jensen-Shannon-Distance", color="yellow",logger=logger)
    plt.plot_kl_divergence(js_divergence_list, end, stepsize, "Jensen-Shannon-Divergence", color="green",logger=logger)

def count_values(value_list, minimum_value, maximum_value, bins):
    """
    Method to count the number of values in a list that fall into each bin of a histogram
    :param value_list: list[float]: list with values to be counted
    :param minimum_value: float: minimum value in global distributions
    :param maximum_value: float: maximum value in global distributions
    :param bins: int: number of bins to be distributed to
    :return: list[float]: counted list
    """
    value_range = maximum_value - minimum_value
    step_size = value_range / bins
    count_list = [0] * bins

    for value in value_list:
        index = int((value - minimum_value) // step_size)
        # Ensure the maximum value is counted in the last bin
        index = min(index, bins-1)
        count_list[index] += 1
    return count_list

def calculate_distributions_dice(p,q):
    """
    Method to calculate the KL divergence in bits, dits, Jensen-Shannon-Distance and Jensen-Shannon-Divergence for dice experiment (not working)
    :param p: base value
    :param q: value to compare to p
    :return: None
    """
    bits,dits=kl_divergence(p,q)
    #plt.plot_kl_divergence_dice(bits)
    jensen_shannon_distance(p,q)
    jensen_shannon_divergence(p,q)

def histogram_from_list(lenght_x, distribution_list, minimum, maximum, num_bins):
    """ transform a list of prediction lists into a list of normalized histograms

    :param lenght_x: int: lenght of the x-axis
    :param distribution_list: list[lists[float]]: list of predictions
    :param minimum: float: minimum value in global distributions
    :param maximum: float: maximum value in global distributions
    :param num_bins: int: number of bins to be distributed to
    :return: list[lists[float]]: list of normalized histograms
    """
    histrogram=[]
    #iterates over all predictions
    for i in range(lenght_x):
        data=[]
        #iterates over all values in predictions
        for lists in distribution_list:
            data.append(lists[i])
        count=count_values(data,minimum,maximum,num_bins)
        count=count/np.sum(count)       #normalizes the count
        histrogram.append(count)
    return histrogram
