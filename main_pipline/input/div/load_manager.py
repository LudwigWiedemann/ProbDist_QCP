import ast

import easygui
import main_pipline.input.div.config_manager as config_manager
import math

def test_inputs(config):
    """
       'time_frame_start': decimal <time_frame_end
       'time_frame_end': decimal
       'n_steps': int > 0
       'time_steps': int > 0
       'future_steps': int >= 0
       'num_samples': int > 0
       'noise_level': decimal >= 0
       'train_test_ratio': decimal >= 0
       'model': str
       'custom_circuit' : bool
       'circuit': str
       'epochs': int > 0
       'batch_size' int > 0
       'learning_rate': decimal > 0
       'loss_function': str
       'steps_to_predict' int >=0
    """

    if not config['time_frame_start'].strip("-").isnumeric:
        raise ValueError("Time_frame_START must be a number")           #filters out only non numeric numbers
    if not config['time_frame_start'].strip("-").isnumeric:
        raise ValueError("Time_frame_END must be a number")             #filters out only non numeric numbers
    elif config['time_frame_end'] <= config['time_frame_start']:
        raise ValueError("Time_frame_END must be greater than time_frame_START")    #filters if end smaller same start
    if not config['n_steps'].isdigit() or int(config['n_steps']) <= 0:
        raise ValueError("N_steps must be a positiv integer")           #filters out non integer numbers, negativ numbers and 0
    if not config['time_steps'].isdigit() or int(config['time_steps']) <= 0:
        raise ValueError("Time_steps must be a positiv integer")        #filters out non integer numbers, negativ numbers and 0
    if not config['future_steps'].isdigit():
        raise ValueError("Future_steps must be a positiv integer")      #filters out non integer numbers and negativ numbers
    if not config['num_samples'].isdigit() or int(config['num_samples']) <= 0:
        raise ValueError("Num_samples must be a positiv integer")        #filters out non integer numbers, negativ numbers and 0
    if not config['noise_level'].isnumeric():
        raise ValueError("Noise_level must be a positiv integer")      #filters out non numeric numbers and negativ numbers
    if not config['train_test_ratio'].isnumeric():
        raise ValueError("Train_test_ratio must be a positiv integer")      #filters out non numeric numbers and negativ numbers
    if config['model'] is None:
        raise ValueError("Model musst be an string representing an existing model") #filters out empty
    if config['custom_circuit'] is None:
        raise ValueError("Custom_circuit cannot be empty")                          #filters out empty
    else:
        try:
            ast.literal_eval(config['custom_circuit'])
        except Exception:
            raise ValueError("Custom_circuit musst be a boolean")               #filters out non-boolean values
    if config['circuit'] is None:
        raise ValueError("Circuit musst be an string representing an existing circuit") #filters out empty
    if not config['epochs'].isdigit():
        raise ValueError("Epochs must be a positiv integer")      #filters out non integer numbers and negativ numbers
    if not config['batch_size'].isdigit():
        raise ValueError("Batch-size must be a positiv integer")      #filters out non integer numbers and negativ numbers
    if not config['learning_rate'].isnumeric() or int(config['learning_rate']) <= 0:
        raise ValueError("Learning_rate must be a positiv number")      #filters out non numeric numbers and negativ numbers
    if config['loss_function'] is None:
        raise ValueError("Loss_function musst be an string representing an existing model") #filters out empty
    if not config['steps_to_predict'].isdigit() or int(config['time_steps']) <= 0:
        raise ValueError("Steps_to_predict must be a positiv integer")        #filters out non integer numbers, negativ numbers and 0


def dialog_load_config():
    button=easygui.buttonbox('Do you want to run with a existing config or create a new one?', 'Load config', ["create custom one", "load exising"])
    match button:
        case "create custom one":
            default_config=[]
        case "load exising":
            filepath = easygui.fileopenbox(msg='Please locate the config.json file',
                                           title='Specify File', default='output\*.PKL',
                                           filetypes='*.json')
            config = config_manager.config_load(filepath)
            default_config = config_manager.to_list(config)
        case _: print("Error: Not a valid input")

    values= [
        'time_frame_start','time_frame_end','n_steps','time_steps','future_steps','num_samples','noise_level','train_test_ratio', 'model','custom_circuit','circuit','epochs','batch_size','learning_rate','loss_function', 'steps_to_predict'
    ]
    msg=""
    while True:
        output = easygui.multenterbox(msg,"Edit Values", values, default_config)
        if output is None:
            break
        output_config= config_manager.config_create(output)
        try:
            test_inputs(output_config)
            break
        except ValueError as e:
            msg=e
            default_config=output
    config_manager.load_from_values(output)
    config_manager.config_save()
    return output_config


