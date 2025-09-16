import os
import pandas as pd
import numpy as np
import mne
import json
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import mne
from PyEMD import EMD
import antropy as ant
import warnings
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from sklearn.model_selection import train_test_split
import random


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level('ERROR')





def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()

def hjorth_parameters(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)
    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0  
    complexity = (np.sqrt(var_d2 / var_d1) / mobility) if var_d1 != 0 and mobility != 0 else 0
    return mobility, complexity

def line_length(signal):
    return np.sum(np.abs(np.diff(signal)))


def extract_nonlinear_features(segment):
    fd = ant.higuchi_fd(segment)  # Fractal Dimension
    return fd



def stats(signal):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    skew_val = 0 if std_val < 1e-8 else skew(signal)
    ptp = np.ptp(signal)
    return [mean_val,std_val,skew_val,ptp]


def feature_extraction(filename):
    filepath=os.path.join("eeg_dataset",filename)
    df=pd.read_csv(filepath)


    df_grouped = df.groupby(df.index // 128).agg(lambda x: list(x))
    df_grouped.reset_index(drop=True, inplace=True)

    df_grouped["PatientId"]=df_grouped["PatientId"].apply(lambda x:set(x).pop())
    df_grouped["Class"]=df_grouped["Class"].apply(lambda x:set(x).pop())

    channels=df_grouped.columns[:2]
    
    for channel in channels:
        df_grouped[f"{channel}_zc"]=df_grouped[channel].apply(lambda x:zero_crossing_rate(np.array(x)))

        df_grouped[[f"{channel}_mob",f"{channel}_com"]]=df_grouped[channel].apply(lambda x:hjorth_parameters(np.array(x))).apply(pd.Series)
        
        df_grouped[f"{channel}_ll"]=df_grouped[channel].apply(lambda x:line_length(np.array(x)))

        df_grouped[[f"{channel}_mean",f"{channel}_std",f"{channel}_skew",f"{channel}_ptp"]]=df_grouped[channel].apply(lambda x:stats(np.array(x))).apply(pd.Series)

        df_grouped[f"{channel}_fd"]=df_grouped[channel].apply(lambda x:extract_nonlinear_features(np.array(x)))
    
    df_grouped.drop(columns=channels,inplace=True)

    df_grouped.to_csv(os.path.join("features_eeg_dataset",filename),index=False)


def raw_dataset_processing(patients:list) :
    for patient in tqdm(patients,desc="Patients: "):
        patient_dfs=[]
        time_dirs=[os.path.join(patient,time) for time in os.listdir(patient) if os.path.isdir(os.path.join(patient,time))]
        for time in time_dirs:
            with open(os.path.join(time,"info.json"),"r") as file:
                d=json.load(file)
            patient_id=d["id"]
            patient_class=1 if "ALS" in patient_id else 0
            scenarios=[os.path.join(time,scenario) for scenario in os.listdir(time) if os.path.isdir(os.path.join(time,scenario))]
            for scenario in scenarios:
                with open(os.path.join(scenario,"scenario.json"),"r") as file:
                    d=json.load(file)
                
                raw = mne.io.read_raw_edf(os.path.join(scenario,"EEG.edf"), preload=True)
                data, _ = raw.get_data(return_times=True)

                df = pd.DataFrame(data=data.T,columns=raw.ch_names)
                df=df[["CP2", "CP6"]]
                df["PatientId"]=patient_id
                df["Class"]=patient_class
                
                patient_dfs.append(df)
            
        eeg_df=pd.concat(patient_dfs,ignore_index=True)
        eeg_df.to_csv(f"eeg_dataset/{patient_id}.csv",index=False)


if __name__ == "__main__":
    dataset_dir="EEGET-ALS Dataset"
    
    eeg_dataset="eeg_dataset"
    features_eeg_dataset="features_eeg_dataset"
    classification_datasets="classification_datasets"
    
    os.makedirs(eeg_dataset,exist_ok=True)
    os.makedirs(features_eeg_dataset,exist_ok=True)
    os.makedirs(classification_datasets,exist_ok=True)

    patients=[os.path.join(dataset_dir,patient) for patient in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,patient))]
    
    # raw_dataset_processing(patients) 
    
    all_datasets=[ds for ds in os.listdir(eeg_dataset) if ds not in os.listdir(features_eeg_dataset)]
    
    with ProcessPoolExecutor(max_workers=7) as executor:
        list(tqdm(executor.map(feature_extraction, all_datasets), total=len(all_datasets)))

    dfs=[]
    for ds in os.listdir(features_eeg_dataset):
        df=pd.read_csv(os.path.join(features_eeg_dataset,ds))
        dfs.append(df)

    eeg_df=pd.concat(dfs)

    eeg_df.to_csv(f"{classification_datasets}/full_dataset.csv",index=False)

    eeg_df=pd.read_csv(f"{classification_datasets}/full_dataset.csv")

    all_patients=eeg_df["PatientId"].unique().tolist()
    
    als_patients=[p for p in all_patients if "ALS" in p]
    control_patients=[p for p in all_patients if "id" in p]

    print("ALS Patients:",len(als_patients))
    print("Control Patients:",len(control_patients))

    for i in range(1,11):
        new_path=f"{classification_datasets}/trial_{i}"
        os.makedirs(new_path,exist_ok=True)
        control_patients=random.sample(control_patients,len(als_patients)*8)

        train_als_patients,temp_als_patients=train_test_split(als_patients,test_size=0.3)
        val_als_patients,test_als_patients=train_test_split(temp_als_patients,test_size=0.5)

        train_control_patients,temp_control_patients=train_test_split(control_patients,test_size=0.3)
        val_control_patients,test_control_patients=train_test_split(temp_control_patients,test_size=0.5)


        train_df=eeg_df[(eeg_df["PatientId"].isin(train_als_patients)) | (eeg_df["PatientId"].isin(train_control_patients))]
        val_df=eeg_df[(eeg_df["PatientId"].isin(val_als_patients)) | (eeg_df["PatientId"].isin(val_control_patients))]
        test_df=eeg_df[(eeg_df["PatientId"].isin(test_als_patients)) | (eeg_df["PatientId"].isin(test_control_patients))]

        train_df=train_df.sample(frac=1)
        val_df=val_df.sample(frac=1)
        test_df=test_df.sample(frac=1)

        train_df.to_csv(f"{new_path}/train.csv",index=False)
        val_df.to_csv(f"{new_path}/val.csv",index=False)
        test_df.to_csv(f"{new_path}/test.csv",index=False)




    
    




