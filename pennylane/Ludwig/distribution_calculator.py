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

def calculate_kl_divergence(list1, list2):
    # Convert the lists to numpy arrays
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # Calculate the KL divergence
    kl_divergence = kl_div(arr1, arr2)

    # Format the result
    result = "KLD: " + str(kl_divergence)

    return result


def kl_div(p, q):
    # Ensure the inputs are numpy arrays

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    # Add a small constant to avoid division by zero
    p = p + 1e-10
    q = q + 1e-10

    # Calculate KL divergence
    divergence = np.sum(rel_entr(p, q))

    return divergence

def calculate_distribution_with_KLD(predictions,datasets,stepsize, start, end):
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
    num_bins = 10            #number of bins the data is divided into
    # Define the bin edges
    bin_edges = np.arange(minimum-1, maximum + 1, (maximum - minimum) / num_bins)
    counts_prediction=[]        #initialize the counts array
    counts_input=[]             #initialize the counts array
    for i in range(lenght_x_prediction):
        data=[]
        for prediction in predictions:
            #print("Prediction: "+str(prediction))
            data.append(prediction[i])
        count, bins = np.histogram(data, bins=bin_edges)
        counts_prediction.append(count/num_predictions)
    #print("COUNTS:"+str(counts))
    for i in range(lenght_x_input):
        if i>=end:
            data=[]
            for dataset in datasets:
                #print("Prediction: "+str(prediction))
                data.append(dataset[i])
            count, bins = np.histogram(data, bins=bin_edges)
            counts_input.append(count/num_inputs)
    kl_divergence_list=[]
    print("KL_length:"+str(len(counts_prediction)))
    print("inputs_length:"+str(len(counts_input)))
    print("start:"+str(start))
    print("end:"+str(end))
    print("END:"+str(lenght_x_input*stepsize))
    for i in range(lenght_x_input-end-15):
        print(i+10)
        kl_divergence_list.append(kl_div(counts_input[i+end-1],counts_prediction[i+end-1]))
        print(f"at x={(i+end)*stepsize}: {kl_divergence_list[i]}")
    plt.plot_kl_divergence(kl_divergence_list, start+end, stepsize, lenght_x_input*stepsize)


