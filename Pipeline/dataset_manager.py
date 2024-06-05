
#TODO adapte the code
dataset_settings = {
    "n_wires": 4,  # Number of wires for the quantum device.
    "min_target_depth": 5,  # Minimum depth of target.
    "max_target_depth": 15,  # Maximum depth of target.
    "size": 150,
    "name": "Test_set_4_150_1_15"
}

def generate_dataset(config, path="../inputs/datasets"):
    # Adjust dtype to complex if you are saving state vectors
    dataset = []
    # TODO add generate_dataset_function
    with open(f"{path}/{config['name']}", 'wb') as f:
        dill.dump(dataset, f)
    print("Finished generating dataset")

def load_dataset(dataset_name):
    # TODO write path
    path = os.path.join(os.path.dirname(__file__), )
    dataset_file = os.path.join(path, dataset_name)
    print(f"Loading dataset from: {dataset_file}", flush=True)
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")
    with open(dataset_file, 'rb') as f:
        dataset = dill.load(f)
    return dataset



