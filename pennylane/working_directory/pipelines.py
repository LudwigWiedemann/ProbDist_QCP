from pennylane import numpy as np
import noise as ns
import training as tr
import plotting as plot

num_training_points = 60
start = 0
stop = 10


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
    training_distributions = prepare_data(num_shots)
    param_list = tr.train_params(training_distributions)
    plot.plot(param_list, training_distributions, tr.f)


def questions():
    num_training_points, start, stop = map(int, input("please enter trainingspoints start stop").split())
    if stop <= start:
        raise ValueError("Stop has to be higher then start")
    if num_training_points < 1:
        raise ValueError("num_training_points has to be a positive integer")
    shots = int(input("Please enter number of shots: "))
    if shots < 1:
        raise ValueError("Number of shots has to be a positive integer")
    runs = input("Please enter the number of runs (default=1) or 'inf': ")
    if not runs == 'inf':
        if int(runs) < 1 or isinstance(runs, int):
            raise ValueError("Shots has to be a positive integer or inf")
    xyz_seed, arythmetic_seed = map(int, input("Please enter seeds for xyz, arythmetic: ").split())
    if xyz_seed == "":
        xyz_seed = "rand"
    if arythmetic_seed == "":
        arythmetic_seed = "rand"
    return num_training_points, start, stop, shots, runs


# while True:
#     try:
#         num_training_points, start, stop, num_shots, runs = questions()
#         break
#     except Exception as e:
#         print(e)

training_inputs = np.linspace(start, stop, num_training_points)
# print("run")
# if runs == "inf":
#     while True:
run(3)
# else:
#     for i in range(int(runs)):
#         run(num_shots)
