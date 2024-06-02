import ast

import pennylane as qml
from pennylane import numpy as np
from autoray.autoray import nothing
from logger import logger

import noise as ns
import training as tr
import plotting as plot
import easygui as eg
import json
import time
import numpy
import save


class ParamGui:

    def __init__(self):
        self.num_training_points = 20
        self.start = 0
        self.stop = 10
        self.training_inputs = np.linspace(self.start, self.stop, self.num_training_points)

    def prepare_data(self, num_shots):
        training_datasets = [[]] * num_shots
        for i in range(num_shots):
            training_datasets[i] = [(x, tr.f(x)) for x in self.training_inputs]
            noise = ns.white(self.num_training_points)
            j = 0
            for x, y in training_datasets[i]:
                y += noise[j]
                j += 1
        return training_datasets

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
        if qbits is not "" and int(float(qbits)) < 1:
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

            output = eg.buttonbox("Choose a Run", "", ["Run random", "Run specific"])

            match output:
                case "Run random":
                    nothing
                case "Run specific":
                    title = "Inputs"
                    msg = "Please enter values, empty fields will be assigned randomly if possible"
                    fields = ["Trainingpoint-count", "Startpoint", "Stoppoint", "Layer-count", "Qbit-count",
                              "Shot-count", "Run-count", "Seed"]
                    default = []
                    while True:
                        try:
                            num_training_points, start, stop, num_layers, num_qbits, shots, runs, seed = eg.multenterbox(
                                msg, title, fields, default)
                            check_inputs_valid(num_training_points, start, stop, num_layers, num_qbits, shots, runs,
                                               seed)
                            break
                        except ValueError as e:
                            print(e)
                            msg = e
                            default = [num_training_points, start, stop, num_layers, num_qbits, shots, runs, seed]

            #[[0, 'X*1'], [0, 'Y+x'], [1, 'X*x'], [1, 'Y-x'], [2, 'X-x'], [2, 'Y-x'], [3, 'Y+x'], [5, 'X**x'], [5, 'Z**x'], [6, 'X*1'], [6, 'Y-x'], [8, 'X+x'], [8, 'Z*x']]


class Pipeline:

    def __init__(self):
        logger.info("Pipeline initialized")
        # read from config.json and get circuit, num_qubits, num_layers
        self.circuit = None
        self.num_qubits = None
        self.num_layers = None
        self.training_iterations = None

    def save(self, num_qbits, num_layers, circuit, rotation_matrix, shots, training_iterations, plot_x_start,
             plot_x_stop, plot_x_step, plot_y_start, plot_y_stop, plot_y_step):
        """
        Saves the circuit and the rotation matrix as well as used config.

        :param num_qbits: Number of qubits
        :param num_layers: Number of layers
        :param circuit The used circuit in the "name_circuit" format and "name_circuit_prob" if shots are involved
        :param rotation_matrix The "weights" used in the circuit
        :param shots: Number of shots
        :param training_iterations: Number of training iterations
        :param plot_x: X axis of the plot
        :param plot_y: Y axis of the plot
        :param plot_points: count of points on x


        """
        differentiationline = "--------------------------------------------------------------------"
        msg = ("{" +
               "\nnum_qbits: " + str(num_qbits) +
               "\nnum_layers: " + str(num_layers) +
               "\ncircuit: " + str(circuit) +
               "\nrotation_matrix: " + str(rotation_matrix) +
               "\n" +
               "\nprobability_mode: " + ("true" if shots > 1 else "false") +
               "\nshots: " + str(shots) +
               "\n" +
               "\ntraining_iterations: " + str(training_iterations) +
               "\n" +
               "\nload_mode: true"
               "\nplot_x_start: " + str(plot_x_start) +
               "\nplot_x_stop: " + str(plot_x_stop) +
               "\nplot_x_step: " + str(plot_x_step) +
               "\nplot_y_start: " + str(plot_y_start) +
               "\nplot_y_stop: " + str(plot_y_stop) +
               "\nplot_y_step: " + str(plot_y_step) +
               "\n}")
        logger.info("Parameter:\n" + differentiationline + "\n" + msg + "\n" + differentiationline)
        filename = save.start_time + "-config"
        jsonlist = {
            "num_qbits: ": str(num_qbits),
            "num_layers: ": str(num_layers),
            "circuit: ": str(circuit),
            "rotation_matrix: ": str(rotation_matrix),
            "probability_mode: ": (True if shots > 1 else False),
            "shots: ": str(shots),
            "training_iterations: ": str(training_iterations),
            "load_mode: ": True,
            "plot_x_start: ": str(plot_x_start),
            "plot_x_stop: ": str(plot_x_stop),
            "plot_x_step: ": str(plot_x_step),
            "plot_y_start: ": str(plot_y_start),
            "plot_y_stop: ": str(plot_y_stop),
            "plot_y_step: ": str(plot_y_step)
        }
        with open(f'../Saves/{filename}' + ".json", "w+") as f:
            json.dump(jsonlist, f, indent=2)

    def load(self, filename):
        """
        Loads the circuit and the rotation matrix as well as optionally used config.
        """
        with open(f'../Saves/{filename}' + ".json", "r") as f:
            data = json.load(f)
            num_qbits = int(data['num_qbits: '])
            num_layers = int(data['num_layers: '])
            circuit= str(data['circuit: '])
            rotation_matrix: [int] = data['rotation_matrix: '].replace(" ","").replace("\n","").strip("[array(").strip("\n").strip(")]").split(",")
            rotation_matrix = [eval(i) for i in rotation_matrix]
            probability_mode= bool(data['probability_mode: '])
            shots= int(data['shots: '])
            training_iterations= int(data['training_iterations: '])
            load_mode= bool(data['load_mode: '])
            plot_x_start= int(data['plot_x_start: '])
            plot_x_stop= int(data['plot_x_stop: '])
            plot_x_step= int(data['plot_x_step: '])
            plot_y_start= int(data['plot_y_start: '])
            plot_y_stop= int(data['plot_y_stop: '])
            plot_y_step= int(data['plot_y_step: '])
        return num_qbits,num_layers,circuit, rotation_matrix, probability_mode, shots, training_iterations, load_mode,plot_x_start, plot_x_stop, plot_x_step, plot_y_start, plot_y_stop, plot_y_step


def run(num_shots):
    save.shots = num_shots
    gui = ParamGui()
    print("run")
    training_distributions = gui.prepare_data(num_shots)
    param_list = tr.train_params(training_distributions)
    plot.plot(param_list, training_distributions, tr.f)
    pip = Pipeline()
    pip.save(save.num_qbits, save.num_layers, save.circuit, save.rotation_matrix, save.shots, save.training_iterations,
             save.plot_x_start, save.plot_x_stop, save.plot_x_step, save.plot_y_start, save.plot_y_stop,
             save.plot_y_step)


print(Pipeline.load(Pipeline,'02.06.2024-04,44,06-config'))

run(1)
