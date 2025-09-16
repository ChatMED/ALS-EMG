import os
import wfdb
import pandas as pd
import numpy as np
import antropy as ant
import pywt
from PyEMD import EMD
from scipy.signal import hilbert
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from scipy import signal as sig
import nolds
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def flatten(matrix):
    res=[]
    for item in matrix:
        if isinstance(item,list) or isinstance(item,np.ndarray):
            res=res+flatten(item)
        else:
            res.append(item)

    return res

def extract_time_features(segment,threshold=0.01):
    diff1 = np.diff(segment)
    diff2 = np.diff(diff1)
    ssc = np.sum((diff1[:-1] * diff1[1:] < 0) & (np.abs(diff2) > threshold)) # Signal Sign Changes
    return [ssc]


def extract_frequency_features(signal_data, fs=23437.5):
    f, psd = sig.periodogram(signal_data, fs)
    
    idx = np.where(f > 0)[0]
    f = f[idx]
    psd = psd[idx]
    
    psd_norm = psd / np.sum(psd)
    
    cumsum_psd = np.cumsum(psd_norm)
    median_freq_idx = np.argmax(cumsum_psd >= 0.5)
    median_freq = f[median_freq_idx]
    
    return [median_freq]

def extract_nonlinear_features(segment,m=2,r=0.2):
    apen = ant.app_entropy(segment)  # Approximate Entropy
    sampEn = ant.sample_entropy(segment)  # Sample Entropy
    fd = ant.higuchi_fd(segment)  # Fractal Dimension

    if r < 1:  # assuming r is meant to be a fraction of std
        r = r * np.std(segment)
        
    return [apen, sampEn, fd]

def extract_psd_features(signal, fs=23437.5):
    f, psd = sig.welch(signal, fs, nperseg=min(2048, len(signal)))

    high_idx = np.logical_and(f > 100, f <= 400)
    
    high_power = np.sum(psd[high_idx])
    
    return [high_power]


def extract_spectral_features(segment,fs=23437.5):
    coefficients, _ = pywt.cwt(segment, np.arange(1, 128), 'morl')
    we=np.sum(coefficients ** 2)  # Wavelength Energy
    return [we]


def detect_muaps(segment, fs=23437.5, threshold_factor=3):
    
    # High-pass filter to remove baseline
    sos = sig.butter(4, 20, 'hp', fs=fs, output='sos')
    filtered = sig.sosfilt(sos, segment)
    
    # Detect peaks (potential MUAPs)
    threshold = threshold_factor * np.std(filtered)
    peaks, _ = sig.find_peaks(np.abs(filtered), height=threshold, distance=int(0.03*fs))
    
    if len(peaks) == 0:
        return [
             0,
             0
        ]
    
    # MUAP amplitudes
    amplitudes = filtered[peaks]
    return [
        len(peaks),
        np.mean(np.abs(amplitudes)),
    ]

def segmentation(signal,fs=23437,window_length=1):
    window_size = int(round(fs * window_length))
    segments = [signal[i:i + window_size] for i in range(0, len(signal), window_size)]
    segments = [s for s in segments if len(s) == window_size]
    return segments

def generate_database(data,file_names):
    
    def classification_func(row):
        return {"C":0,"M":2, "A":1}[row["Diagnosis"]]

    df_data=[
        {
            "Diagnosis":file_names[i][5],
            "Patient":file_names[i][5:8],
            "Signal":data[i]
        } for i in range(0,len(data))
    ]

    df=pd.DataFrame(df_data)

    df["Class"]=df.apply(classification_func,axis=1)
    df.drop(columns=["Diagnosis"],inplace=True)
    return df

def parse_binary_files(database_dir):
    signal_data_list=[]
    record_name_list=[]

    hea_files=[f for f in os.listdir(database_dir) if f.endswith(".hea")]

    for file in hea_files:
        record_name=os.path.splitext(file)[0]
        record_path=os.path.join(database_dir,record_name)
        record=wfdb.rdrecord(record_path)

        signal_data=np.array(record.p_signal).flatten()
        
        signal_data_list.append(signal_data)
        record_name_list.append(record_name)

    signal_data_array=np.stack(signal_data_list)

    data = list(zip(record_name_list, signal_data_array))
    sorted_data = sorted(data, key=lambda x: x[0])

    record_name_list=[x[0] for x in sorted_data]
    signal_data_array=[x[1] for x in sorted_data]

    return signal_data_array,record_name_list

def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def feature_extraction(file_df):
    # filepath=os.path.join("als_datasets/patient_datasets",filename)
    # file_df=pd.read_csv(filepath)
    
    filename=file_df["Patient"].unique().tolist()[0]
    file_df["Signal_Segment"]=file_df["Signal"].apply(lambda sig:segmentation(sig,window_length=0.25))

    patient_df=pd.DataFrame(columns=["Patient","Class","Interval"])

    for i,row in file_df.iterrows():
        for seg in row["Signal_Segment"]:
            new_row=[row["Patient"],row["Class"],seg]
            patient_df.loc[len(patient_df)]=new_row

   
    patient_df["Interval"]=patient_df["Interval"].apply(lambda seg:[s for s in seg if not np.isnan(s)])
    patient_df["To_Drop"]=patient_df["Interval"].apply(lambda seg:len(seg) == 5859) # interval size before removing nan values
    patient_df=patient_df[patient_df["To_Drop"] == True].drop(columns=["To_Drop"])
    
    patient_df[["SSC"]]=patient_df["Interval"].apply(lambda seg:extract_time_features(np.array(seg))).apply(pd.Series)
    
    patient_df[["MDF"]]=patient_df["Interval"].apply(lambda seg:extract_frequency_features(np.array(seg))).apply(pd.Series)
    
    patient_df[['high_power']]=patient_df["Interval"].apply(lambda seg:extract_psd_features(np.array(seg))).apply(pd.Series)
    
    patient_df[["APEN", "SampEn", "FracDim",]] = patient_df["Interval"].apply(lambda seg:extract_nonlinear_features(np.array(seg))).apply(pd.Series)
    
    patient_df[["WE"]]=patient_df["Interval"].apply(lambda seg:extract_spectral_features(np.array(seg))).apply(pd.Series)

    patient_df[["muap_1","muap_2"]]=patient_df["Interval"].apply(lambda seg:detect_muaps(np.array(seg))).apply(pd.Series)

    patient_df.drop(columns=["Interval"],inplace=True)

    patient_df.to_csv(f"als_datasets/features_emg_dataset/{filename}.csv",index=False)

if __name__ == "__main__":
    dataset_directory="als_datasets"
    emglab_dataset=os.path.join(dataset_directory,"emglab_dataset_bb")
    patient_datasets=os.path.join(dataset_directory,"patient_datasets")
    features_emg_dataset=os.path.join(dataset_directory,"features_emg_dataset")

    control_vs_als_dataset=os.path.join(dataset_directory,"control_vs_als_dataset")
    control_vs_myopathy_dataset=os.path.join(dataset_directory,"control_vs_myopathy_dataset")
    als_vs_myopathy_dataset=os.path.join(dataset_directory,"als_vs_myopathy_dataset")
    control_vs_als_vs_myopathy_dataset=os.path.join(dataset_directory,"control_vs_als_vs_myopathy_dataset")

    als_vs_control_and_myopathy_dataset=os.path.join(dataset_directory,"als_vs_control_and_myopathy_dataset")
    control_vs_als_and_myopathy_dataset=os.path.join(dataset_directory,"control_vs_als_and_myopathy_dataset")
    myopathy_vs_control_and_als_dataset=os.path.join(dataset_directory,"myopathy_vs_control_and_als_dataset")

    create_folder_if_not_exist(patient_datasets)
    create_folder_if_not_exist(features_emg_dataset)

    create_folder_if_not_exist(control_vs_als_dataset)
    create_folder_if_not_exist(control_vs_myopathy_dataset)
    create_folder_if_not_exist(als_vs_myopathy_dataset)
    create_folder_if_not_exist(control_vs_als_vs_myopathy_dataset)
    create_folder_if_not_exist(als_vs_control_and_myopathy_dataset)
    create_folder_if_not_exist(control_vs_als_and_myopathy_dataset)
    create_folder_if_not_exist(myopathy_vs_control_and_als_dataset)

    signal_data_array,record_name_list=parse_binary_files(emglab_dataset)
    df=generate_database(signal_data_array,record_name_list)

    all_datasets=[df[df["Patient"] == patient] for patient in df["Patient"].unique().tolist()]
    
    with ProcessPoolExecutor(max_workers=7) as executor:
        list(tqdm(executor.map(feature_extraction, all_datasets), total=len(all_datasets)))


    dfs=[]
    for ds in os.listdir(features_emg_dataset):
        df=pd.read_csv(os.path.join(features_emg_dataset,ds))
        dfs.append(df)

    emg_df=pd.concat(dfs)

    emg_df.to_csv(f"{dataset_directory}/full_dataset.csv",index=False)

    emg_df=pd.read_csv("als_datasets/full_dataset.csv")

    all_patients=emg_df["Patient"].unique().tolist()

    control_patients=[p for p in all_patients if "C" in p]
    als_patients=[p for p in all_patients if "A" in p]
    myopathy_patients=[p for p in all_patients if "M" in p]

    train_control_patients,temp_control_patients=train_test_split(control_patients,test_size=0.3)
    val_control_patients,test_control_patients=train_test_split(temp_control_patients,test_size=0.5)

    train_als_patients,temp_als_patients=train_test_split(als_patients,test_size=0.3)
    val_als_patients,test_als_patients=train_test_split(temp_als_patients,test_size=0.5)

    train_myopathy_patients,temp_myopathy_patients=train_test_split(myopathy_patients,test_size=0.3)
    val_myopathy_patients,test_myopathy_patients=train_test_split(temp_myopathy_patients,test_size=0.5)


    train_control_vs_als_df=emg_df[(emg_df["Patient"].isin(train_als_patients)) | (emg_df["Patient"].isin(train_control_patients))]
    val_control_vs_als_df=emg_df[(emg_df["Patient"].isin(val_als_patients)) | (emg_df["Patient"].isin(val_control_patients))]
    test_control_vs_als_df=emg_df[(emg_df["Patient"].isin(test_als_patients)) | (emg_df["Patient"].isin(test_control_patients))]

    train_control_vs_als_df=train_control_vs_als_df.sample(frac=1)
    val_control_vs_als_df=val_control_vs_als_df.sample(frac=1)
    test_control_vs_als_df=test_control_vs_als_df.sample(frac=1)

    train_control_vs_als_df.to_csv(f"{control_vs_als_dataset}/train.csv",index=False)
    val_control_vs_als_df.to_csv(f"{control_vs_als_dataset}/val.csv",index=False)
    test_control_vs_als_df.to_csv(f"{control_vs_als_dataset}/test.csv",index=False)


    train_control_vs_myopathy_df=emg_df[(emg_df["Patient"].isin(train_myopathy_patients)) | (emg_df["Patient"].isin(train_control_patients))]
    val_control_vs_myopathy_df=emg_df[(emg_df["Patient"].isin(val_myopathy_patients)) | (emg_df["Patient"].isin(val_control_patients))]
    test_control_vs_myopathy_df=emg_df[(emg_df["Patient"].isin(test_myopathy_patients)) | (emg_df["Patient"].isin(test_control_patients))]

    train_control_vs_myopathy_df["Class"]=train_control_vs_myopathy_df["Class"].apply(lambda x:1 if x == 2 else x)
    val_control_vs_myopathy_df["Class"]=val_control_vs_myopathy_df["Class"].apply(lambda x:1 if x == 2 else x)
    test_control_vs_myopathy_df["Class"]=test_control_vs_myopathy_df["Class"].apply(lambda x:1 if x == 2 else x)
    
    train_control_vs_myopathy_df=train_control_vs_myopathy_df.sample(frac=1)
    val_control_vs_myopathy_df=val_control_vs_myopathy_df.sample(frac=1)
    test_control_vs_myopathy_df=test_control_vs_myopathy_df.sample(frac=1)

    train_control_vs_myopathy_df.to_csv(f"{control_vs_myopathy_dataset}/train.csv",index=False)
    val_control_vs_myopathy_df.to_csv(f"{control_vs_myopathy_dataset}/val.csv",index=False)
    test_control_vs_myopathy_df.to_csv(f"{control_vs_myopathy_dataset}/test.csv",index=False)


    train_als_vs_myopathy_df=emg_df[(emg_df["Patient"].isin(train_myopathy_patients)) | (emg_df["Patient"].isin(train_als_patients))]
    val_als_vs_myopathy_df=emg_df[(emg_df["Patient"].isin(val_myopathy_patients)) | (emg_df["Patient"].isin(val_als_patients))]
    test_als_vs_myopathy_df=emg_df[(emg_df["Patient"].isin(test_myopathy_patients)) | (emg_df["Patient"].isin(test_als_patients))]

    train_als_vs_myopathy_df["Class"]=train_als_vs_myopathy_df["Class"].apply(lambda x:0 if x == 2 else x)
    val_als_vs_myopathy_df["Class"]=val_als_vs_myopathy_df["Class"].apply(lambda x:0 if x == 2 else x)
    test_als_vs_myopathy_df["Class"]=test_als_vs_myopathy_df["Class"].apply(lambda x:0 if x == 2 else x)
   
    train_als_vs_myopathy_df=train_als_vs_myopathy_df.sample(frac=1)
    val_als_vs_myopathy_df=val_als_vs_myopathy_df.sample(frac=1)
    test_als_vs_myopathy_df=test_als_vs_myopathy_df.sample(frac=1)

    train_als_vs_myopathy_df.to_csv(f"{als_vs_myopathy_dataset}/train.csv",index=False)
    val_als_vs_myopathy_df.to_csv(f"{als_vs_myopathy_dataset}/val.csv",index=False)
    test_als_vs_myopathy_df.to_csv(f"{als_vs_myopathy_dataset}/test.csv",index=False)


    train_control_vs_als_vs_myopathy_df=emg_df[(emg_df["Patient"].isin(train_myopathy_patients)) | (emg_df["Patient"].isin(train_als_patients)) | (emg_df["Patient"].isin(train_control_patients))]
    val_control_vs_als_vs_myopathy_df=emg_df[(emg_df["Patient"].isin(val_myopathy_patients)) | (emg_df["Patient"].isin(val_als_patients)) | (emg_df["Patient"].isin(val_control_patients))]
    test_control_vs_als_vs_myopathy_df=emg_df[(emg_df["Patient"].isin(test_myopathy_patients)) | (emg_df["Patient"].isin(test_als_patients)) | (emg_df["Patient"].isin(test_control_patients))]

    train_control_vs_als_vs_myopathy_df=train_control_vs_als_vs_myopathy_df.sample(frac=1)
    val_control_vs_als_vs_myopathy_df=val_control_vs_als_vs_myopathy_df.sample(frac=1)
    test_control_vs_als_vs_myopathy_df=test_control_vs_als_vs_myopathy_df.sample(frac=1)

    train_control_vs_als_vs_myopathy_df.to_csv(f"{control_vs_als_vs_myopathy_dataset}/train.csv",index=False)
    val_control_vs_als_vs_myopathy_df.to_csv(f"{control_vs_als_vs_myopathy_dataset}/val.csv",index=False)
    test_control_vs_als_vs_myopathy_df.to_csv(f"{control_vs_als_vs_myopathy_dataset}/test.csv",index=False)


    train_als_vs_control_and_myopathy_df=emg_df[(emg_df["Patient"].isin(train_myopathy_patients)) | (emg_df["Patient"].isin(train_als_patients)) | (emg_df["Patient"].isin(train_control_patients))]
    val_als_vs_control_and_myopathy_df=emg_df[(emg_df["Patient"].isin(val_myopathy_patients)) | (emg_df["Patient"].isin(val_als_patients)) | (emg_df["Patient"].isin(val_control_patients))]
    test_als_vs_control_and_myopathy_df=emg_df[(emg_df["Patient"].isin(test_myopathy_patients)) | (emg_df["Patient"].isin(test_als_patients)) | (emg_df["Patient"].isin(test_control_patients))]
    
    train_als_vs_control_and_myopathy_df["Class"]=train_als_vs_control_and_myopathy_df["Class"].apply(lambda x:0 if x == 2 else x)
    val_als_vs_control_and_myopathy_df["Class"]=val_als_vs_control_and_myopathy_df["Class"].apply(lambda x:0 if x == 2 else x)
    test_als_vs_control_and_myopathy_df["Class"]=test_als_vs_control_and_myopathy_df["Class"].apply(lambda x:0 if x == 2 else x)

    train_als_vs_control_and_myopathy_df=train_als_vs_control_and_myopathy_df.sample(frac=1)
    val_als_vs_control_and_myopathy_df=val_als_vs_control_and_myopathy_df.sample(frac=1)
    test_als_vs_control_and_myopathy_df=test_als_vs_control_and_myopathy_df.sample(frac=1)

    train_als_vs_control_and_myopathy_df.to_csv(f"{als_vs_control_and_myopathy_dataset}/train.csv",index=False)
    val_als_vs_control_and_myopathy_df.to_csv(f"{als_vs_control_and_myopathy_dataset}/val.csv",index=False)
    test_als_vs_control_and_myopathy_df.to_csv(f"{als_vs_control_and_myopathy_dataset}/test.csv",index=False)

    
    train_control_vs_als_and_myopathy_df=emg_df[(emg_df["Patient"].isin(train_myopathy_patients)) | (emg_df["Patient"].isin(train_als_patients)) | (emg_df["Patient"].isin(train_control_patients))]
    val_control_vs_als_and_myopathy_df=emg_df[(emg_df["Patient"].isin(val_myopathy_patients)) | (emg_df["Patient"].isin(val_als_patients)) | (emg_df["Patient"].isin(val_control_patients))]
    test_control_vs_als_and_myopathy_df=emg_df[(emg_df["Patient"].isin(test_myopathy_patients)) | (emg_df["Patient"].isin(test_als_patients)) | (emg_df["Patient"].isin(test_control_patients))]

    train_control_vs_als_and_myopathy_df["Class"]=train_control_vs_als_and_myopathy_df["Class"].apply(lambda x:1 if x == 2 else x)
    val_control_vs_als_and_myopathy_df["Class"]=val_control_vs_als_and_myopathy_df["Class"].apply(lambda x:1 if x == 2 else x)
    test_control_vs_als_and_myopathy_df["Class"]=test_control_vs_als_and_myopathy_df["Class"].apply(lambda x:1 if x == 2 else x)

    train_control_vs_als_and_myopathy_df=train_control_vs_als_and_myopathy_df.sample(frac=1)
    val_control_vs_als_and_myopathy_df=val_control_vs_als_and_myopathy_df.sample(frac=1)
    test_control_vs_als_and_myopathy_df=test_control_vs_als_and_myopathy_df.sample(frac=1)

    train_control_vs_als_and_myopathy_df.to_csv(f"{control_vs_als_and_myopathy_dataset}/train.csv",index=False)
    val_control_vs_als_and_myopathy_df.to_csv(f"{control_vs_als_and_myopathy_dataset}/val.csv",index=False)
    test_control_vs_als_and_myopathy_df.to_csv(f"{control_vs_als_and_myopathy_dataset}/test.csv",index=False)


    train_myopathy_vs_control_and_als_df=emg_df[(emg_df["Patient"].isin(train_myopathy_patients)) | (emg_df["Patient"].isin(train_als_patients)) | (emg_df["Patient"].isin(train_control_patients))]
    val_myopathy_vs_control_and_als_df=emg_df[(emg_df["Patient"].isin(val_myopathy_patients)) | (emg_df["Patient"].isin(val_als_patients)) | (emg_df["Patient"].isin(val_control_patients))]
    test_myopathy_vs_control_and_als_df=emg_df[(emg_df["Patient"].isin(test_myopathy_patients)) | (emg_df["Patient"].isin(test_als_patients)) | (emg_df["Patient"].isin(test_control_patients))]

    train_myopathy_vs_control_and_als_df["Class"]=train_myopathy_vs_control_and_als_df["Class"].apply(lambda x:0 if x == 1 else 1 if x== 2 else 0)
    val_myopathy_vs_control_and_als_df["Class"]=val_myopathy_vs_control_and_als_df["Class"].apply(lambda x:0 if x == 1 else 1 if x== 2 else 0)
    test_myopathy_vs_control_and_als_df["Class"]=test_myopathy_vs_control_and_als_df["Class"].apply(lambda x:0 if x == 1 else 1 if x== 2 else 0)

    train_myopathy_vs_control_and_als_df=train_myopathy_vs_control_and_als_df.sample(frac=1)
    val_myopathy_vs_control_and_als_df=val_myopathy_vs_control_and_als_df.sample(frac=1)
    test_myopathy_vs_control_and_als_df=test_myopathy_vs_control_and_als_df.sample(frac=1)

    train_myopathy_vs_control_and_als_df.to_csv(f"{myopathy_vs_control_and_als_dataset}/train.csv",index=False)
    val_myopathy_vs_control_and_als_df.to_csv(f"{myopathy_vs_control_and_als_dataset}/val.csv",index=False)
    test_myopathy_vs_control_and_als_df.to_csv(f"{myopathy_vs_control_and_als_dataset}/test.csv",index=False)
    
