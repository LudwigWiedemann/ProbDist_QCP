import circuit as cir
import matplotlib.pyplot as plt
from logger import logger
import numpy as np
import save


def plot(param_list, distributions, f, x_start, x_stop, x_step, y_start, y_stop, y_step, circuit):

    x_axis = np.linspace(x_start, x_stop, x_step)

    firstparam = True
    for params in param_list:
        match circuit:
            case "run_circuit": predicted_outputs = [cir.run_circuit(params, x) for x in x_axis]
            case "ry_circuit": predicted_outputs = [cir.Circuit.ry_circuit(params, x) for x in x_axis]
            case "entangling_circuit" : predicted_outputs= [cir.Circuit.entangling_circuit(params, x) for x in x_axis]
            case _ :
                    logger.ERROR("Circuit not found!")
                    raise InterruptedError("Circuit not found!")
        if firstparam:
            plt.plot(x_axis, predicted_outputs, 'g--', label="Predicted Sin", alpha=0.1)
        else:
            plt.plot(x_axis, predicted_outputs, 'g--', alpha=0.1)
        firstparam = False


    firstdist = True
    for dist in distributions:
        training_x = [data[0] for data in dist]
        training_y = [data[1] for data in dist]
        if firstdist:
            plt.scatter(training_x, training_y, s=5, label="data points", alpha=0.1)
        else:
            plt.scatter(training_x, training_y, s=5, alpha=0.1)
        firstdist = False

    true_outputs = f(x_axis)
    plt.plot(x_axis, true_outputs, label="Actual f(x)", alpha=0.3)

    plt.ylim(y_start, y_stop)
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(str(len(distributions))+" Shots")
    plt.savefig("../Logger/" + save.start_time + "-plot.png")
    plt.show()
    save.plot_x_start=x_start
    save.plot_x_stop=x_stop
    save.plot_x_step=x_step
    save.plot_y_start=y_start
    save.plot_y_stop=y_stop
    save.plot_y_step=y_step
