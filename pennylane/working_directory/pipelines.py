
import numpy as np
import noise as ns
import training as tr
import plotting as plot
import easygui as eg
import json

num_training_points = 20
start = 0
stop = 10
training_inputs = np.linspace(start, stop, num_training_points)



def prepare_data(num_shots):
    training_datasets = [[]] * num_shots
    for i in range(num_shots):
        training_datasets[i] = [(x, tr.f(x)) for x in training_inputs]
        noise = ns.white(num_training_points)
        j = 0
        for x, y in training_datasets[i]:
            y += noise[j]
            j += 1
    return training_datasets


def run(num_shots):
    print("run")
    training_distributions = prepare_data(num_shots)
    param_list = tr.train_params(training_distributions)
    plot.plot(param_list, training_distributions, tr.f, [[0, 'Y*x']])




def check_inputs_valid(training_points, startpoint, stoppoint, layers, qbits, shotcount, runcountz, seed):
    if training_points is not "":
        training_points = int(float(training_points))
        if int(float(training_points)) < 1:
            raise ValueError("Trainingpoint-count has to be a positive integer")
    if startpoint is "":
        raise ValueError("Start cannot be None")
    if stoppoint is "":
        raise ValueError("Stop cannot be None")
    elif startpoint >= stoppoint:
        raise ValueError("Start must be less than Stop")
    if qbits is not "" and int(float(layers)) < 1:
        raise ValueError("Layer-count has to be a positive integer or empty")
    if qbits is not "" and int(float(qbits)) < 1 :
        raise ValueError("Qbit-count has to be a positive integer or empty")
    if shotcount is "":
        raise ValueError("Shot-count cannot be None")
    elif int(float(shotcount)) < 1:
        raise ValueError("Shot-count has to be a positive integer")
    if runcountz is not "" and int(float(runcountz)) < 1:
        raise ValueError("Run-count has to be a positive integer or empty for infinite runs")
    seed_list = json.loads(seed)
    for elements in seed_list:
        layer, cir = elements.split
        xyz, arithmetic = cir[0], cir[1:]
        if layer is "":
            raise ValueError("Seed is not correctly formated at layer")
        elif int(float(layer)) < 0 or layer >= layers:
            raise ValueError("Seed has to be between 1 and num_layers-1")
        if xyz is "" or xyz not in ["X", "Y", "Z"]:
            raise ValueError("Seed is not correctly formated at XYZ")
        if arithmetic is "" or arithmetic not in ["*1", "+x", "-x", "*x", "**x"]:
            raise ValueError("Seed is not correctly formated at arithmetic")

output = eg.buttonbox("Choose a Run","",["Run random", "Run specific"])

match output:
    case "Run random": nothing
    case "Run specific":
        title= "Inputs"
        msg= "Please enter values, empty fields will be assigned randomly if possible"
        fields=["Trainingpoint-count", "Startpoint", "Stoppoint", "Layer-count", "Qbit-count", "Shot-count", "Run-count", "Seed"]
        default=[]
        while True:
            try:
                num_training_points, start, stop, num_layers, num_qbits, shots, runs, seed = eg.multenterbox(msg, title, fields, default)
                check_inputs_valid(num_training_points, start, stop,num_layers, num_qbits, shots, runs, seed)
                break
            except ValueError as e:
                print(e)
                msg=e
                default= [num_training_points, start, stop, num_layers, num_qbits, shots, runs, seed]

#[[0, 'X*1'], [0, 'Y+x'], [1, 'X*x'], [1, 'Y-x'], [2, 'X-x'], [2, 'Y-x'], [3, 'Y+x'], [5, 'X**x'], [5, 'Z**x'], [6, 'X*1'], [6, 'Y-x'], [8, 'X+x'], [8, 'Z*x']]