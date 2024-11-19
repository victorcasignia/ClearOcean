import yaml
from tempfile import NamedTemporaryFile
import os
import subprocess
import time
import shutil

dataset_root = '/dataset_test/'

def load_yaml_template(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, filepath):
    with open(filepath, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def modify_yaml_config(dir_name, config_data, dataset_name, dataset_type, checkpoint_loc, size):
    # Modify general settings
    config_data['name'] = f"inference_test_{dir_name}_{dataset_name}_{dataset_type}_{size}"
    
    # Modify dataset settings for 'train' and 'val'
    for key in ['train', 'val']:
        config_data['datasets'][key]['dataset_names'] = [dataset_name]
        config_data['datasets'][key]['type'] = "Test_UW_Dataset_Preloaded"
        config_data['datasets'][key]['test_type'] = f"{dataset_type}_{size}"
        config_data['datasets'][key]['crop_size'] = int(size)

    # Modify checkpoint location in the path
    config_data['path']['pretrain_network_g'] = checkpoint_loc
    config_data['train']['total_iter'] = 0
    config_data['val']['only_save_sr'] = True


    return config_data

def run_inference():
    # Load template once
    clear_ocean_experiments = '/mnt/f/ClearOcean_experiments_20241104/'
    for dir_name in os.listdir(clear_ocean_experiments):
        dir_path = os.path.join(clear_ocean_experiments, dir_name)
        if os.path.isdir(dir_path):
            yaml_file_path = os.path.join(dir_path, f"{dir_name}.yaml")
            if os.path.isfile(yaml_file_path):
                template_path = yaml_file_path
                checkpoint_loc = os.path.join(dir_path, "models", "net_g_latest.pth")
                
                dataset_names = ['lsui', 'uieb']
                sizes = ['128', '256']
                dataset_types = ['center_crop', 'random_crop']

                # Load YAML template
                config_template = load_yaml_template(template_path)

                for dataset_name in dataset_names:
                    for size in sizes:
                        for dataset_type in dataset_types:
                            # Modify the template with the current values
                            modified_config = modify_yaml_config(dir_name, config_template.copy(), dataset_name, dataset_type, checkpoint_loc, size)
                            
                            # Save modified config to a temporary YAML file
                            with NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config:
                                save_yaml(modified_config, temp_config.name)
                            
                            # Run the training script with the modified config
                            subprocess.call(["python", "pydiff/train.py", "-opt", temp_config.name])
                            time.sleep(2)  # Optional delay between runs

                            # Move results
                            experiment_path = f"/mnt/f/CS298/PyDIff/experiments/inference_test_{dir_name}_{dataset_name}_{dataset_type}_{size}/visualization"
                            target_path = f"/mnt/f/dataset_test/{dataset_type}_{size}/{dataset_name}/{dir_name}"
                            os.makedirs(target_path, exist_ok=True)

                                                        
                            for item in os.listdir(experiment_path):
                                source = os.path.join(experiment_path, item)
                                shutil.move(source, target_path)
                                
                            # Clean up temporary file
                            os.remove(temp_config.name)

if __name__ == '__main__':
    run_inference()
