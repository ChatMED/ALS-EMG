import subprocess
import os
os.makedirs("results",exist_ok=True)

datasets_eeg = ["trial_1","trial_2","trial_3","trial_4","trial_5","trial_6","trial_7","trial_8","trial_9","trial_10"]
datasets_emg=['control_vs_als_dataset', 'als_vs_myopathy_dataset', 'control_vs_myopathy_dataset', 'control_vs_als_vs_myopathy_dataset', 'myopathy_vs_control_and_als_dataset', 'control_vs_als_and_myopathy_dataset', 'als_vs_control_and_myopathy_dataset']

datasets_emg=['myopathy_vs_control_and_als_dataset', 'control_vs_als_and_myopathy_dataset', 'als_vs_control_and_myopathy_dataset']
for dataset in datasets_emg:
    print(f"\nStarting fine-tuning on: {dataset}")
    result = subprocess.run(["python", "model_finetuning_script.py", dataset,"EMG","128"])
    
    if result.returncode != 0:
        print(f"Error running script for {dataset}. Exiting.")
        break  

    print(f"Finished fine-tuning on: {dataset}")

datasets_eeg = ["trial_8","trial_9","trial_10"]
for dataset in datasets_eeg:
    print(f"\nStarting fine-tuning on: {dataset}")
    result = subprocess.run(["python", "model_finetuning_script.py", dataset,"EEG","256"])
    
    if result.returncode != 0:
        print(f"Error running script for {dataset}. Exiting.")
        break  

    print(f"Finished fine-tuning on: {dataset}")


