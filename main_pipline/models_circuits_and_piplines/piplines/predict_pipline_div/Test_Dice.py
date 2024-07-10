import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import distribution_calculator as dc


num_bits = 20  # Adjust this value for higher or lower precision in the random number generation
dev = qml.device("default.qubit", wires=num_bits, shots=1)

@qml.qnode(dev)
def generate_random_bits():
    '''Generate random bits using the Hadamard gate and measure the result in the Z basis.'''
    for i in range(num_bits):
        qml.Hadamard(wires=i)
    return [qml.sample(qml.PauliZ(wires=i)) for i in range(num_bits)]

def bits_to_float(bits):
    '''Convert a list of bits to a float between -1 and 1.'''
    binary_fraction = sum(bit * (1/2)**(i+1) for i, bit in enumerate(bits))
    return 2 * binary_fraction - 1

def get_keys_from_tensor(tensorlist):
    '''Get the keys(names) from a dictionary of tensors'''
    keys=[]
    for key, tensor in tensorlist.items():
        keys.append(key)
    return keys
def map_random_value_to_splits(random_float, split_points, normalized_input_counts):
    '''Map the random float to a specific interval based on the split points'''
    keys=get_keys_from_tensor(normalized_input_counts)
    # Iterate through each split point to find where the random float falls
    for i in range(len(split_points) - 1):
        if split_points[i] <= random_float < split_points[i + 1]:
            return keys[i]
    # Handle edge case where random_float is exactly 1
    if random_float == 1:
        return len(split_points) - 2
    return None

def ggt(a, b):
    '''Calculate the greatest common divisor of two numbers using the Euclidean algorithm.'''
    while b:
        a, b = b, a % b
    return a

def ggt_multiple(values):
    '''Calculate the greatest common divisor of multiple numbers.'''
    if len(values) == 1:
        return values[0]
    current_gcd = values[0]
    for value in values[1:]:
        current_gcd = ggt(current_gcd, value)
    return current_gcd


def generate_random_number_list():
    '''Generate a list of random floats between -1 and 1.'''
    rand = []
    for i in range(100):
        random_bits = generate_random_bits()
        adjusted_bits = [(bit + 1) / 2 for sublist in random_bits for bit in np.atleast_1d(sublist)] # Adjust bits to be 0 or 1
        random_float = bits_to_float(adjusted_bits) # Convert bits to a float between -1 and 1
        #print("Random float between -1 and 1:", random_float)
        rand.append(random_float)
    return rand


def count_values(values):
    '''Count the number of occurrences of each unique value in a list.'''
    value_counts = {value: values.count(value) for value in set(values)}
    return value_counts

def normalize(value_list):
    '''Normalize a list of values to sum up to 1.'''
    # Calculate the total number of elements in Inputs
    total = sum(value_list.values())

    return {value: count / total for value, count in value_list.items()}

def map_random_values_to_splits(random_float_list, adjusted_split_points, normalized_value_counts):
    randsplitted = []
    print("Random Floats:", random_float_list)
    print("Adjusted Split Points:", adjusted_split_points)
    print("Normalized Value Counts:", normalized_value_counts)
    for random_float in random_float_list:
        mapped_index = map_random_value_to_splits(random_float, adjusted_split_points,normalized_value_counts)
        print(f"Random float {random_float} is mapped to interval {mapped_index}")
        randsplitted.append(mapped_index)
    return randsplitted


#inputs=["Josef", "Ina", "Simon", "Josef", "Josef", "Ina"]
inputs=[2,3,4,5,6,7, 3,4,5,6,7,8, 3,4,5,6,7,8,9, 4,5,6,7,8,9,10, 5,6,7,8,9,10,11, 6,7,8,9,10,11,12]
inputs.sort()

# Count each unique value in Inputs to a list
value_counts = count_values(inputs)

# Normalize each count
normalized_value_counts = normalize(value_counts)

# Calculate the GCD of the normalized value counts
ggt_size = ggt_multiple(list(normalized_value_counts.values()))

# Determine the total number of parts
total_parts = int(1 / ggt_size) #defines the number of Bins used

# Calculate the size of each part
part_size = 2 / total_parts

# Calculate the cumulative sum of the normalized counts
cumulative_sums = np.cumsum(list(normalized_value_counts.values()))

# Adjust the split points based on the cumulative sums
# The first split point is always -1, and the last is always 1
adjusted_split_points = [-1] + [2 * cumsum - 1 for cumsum in cumulative_sums[:-1]] + [1]

#print("Adjusted Split Points:", adjusted_split_points)

# Generate a list of random floats between -1 and 1
rand=generate_random_number_list()
randsplitted = map_random_values_to_splits(rand, adjusted_split_points, normalized_value_counts)
randsplitted.sort()

#Count each unique value in randsplitted
output_counts = count_values(randsplitted)

#Normalize each count
normalized_output_counts = normalize(output_counts)

print("Normalized Value Counts:", normalized_value_counts)
print("Normalized Output Counts:", normalized_output_counts)

# Prepare the data
keys = sorted(set(normalized_value_counts.keys()) | set(normalized_output_counts.keys()))
normalized_value_counts_list = [normalized_value_counts.get(key, 0) for key in keys]
normalized_output_counts_list = [normalized_output_counts.get(key, 0) for key in keys]

# Plotting
x = range(len(keys))  # X-axis points
plt.bar(x, normalized_value_counts_list, width=0.4, label='Normalized Value Counts', align='center')
plt.bar(x, normalized_output_counts_list, width=0.4, label='Normalized Output Counts', align='edge')

# Adding details
plt.xlabel('Unique Values/Bins')
plt.ylabel('Normalized Counts')
plt.title('Comparison of Normalized Value Counts and Normalized Output Counts')
plt.xticks(x, keys, rotation='vertical')  # Set x-ticks to be the keys, rotate for readability
plt.legend()

# Display the plot
plt.tight_layout()  # Adjust layout to not cut off labels
plt.show()

#dc.calculate_distributions_dice(normalized_output_counts,normalized_output_counts)