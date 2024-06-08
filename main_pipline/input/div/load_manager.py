import easygui
import config_manager

button=easygui.buttonbox('Do you want to run with a existing config or create a new one?', 'Load config', ["create custom one", "load exising"])

match button:
    case "create custom one":
        default_config=[]
    case "load exising":
        filepath = easygui.fileopenbox(msg='Please locate the config .json file',
                                       title='Specify File', default='output\*.PKL',
                                       filetypes='*.json')
        config = config_manager.config_load(filepath)
        default_config = [config['time_frame_start'], config['time_frame_end'], config['n_steps'], config['time_steps'], config['future_steps'], config['num_samples'], config['noise_level'], config['train_test_ratio']]
    case _: print("Error: Not a valid input")

values= [
    'time_frame_start','time_frame_end','n_steps','time_steps','future_steps','num_samples','noise_level','train_test_ratio',
]
msg=""
easygui.multenterbox(msg,"Edit Values", values, default_config)
config_manager.config_save()



def test_inputs(config):
    None
