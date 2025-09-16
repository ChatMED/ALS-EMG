import os
from typing import Callable
import evaluate
import numpy as np
from sklearn.metrics import roc_auc_score


base_path = "."


def _als_mapping_function(example: dict) -> int:
    return 1 if example["completion"] == "ALS" else 0


def _control_mapping_function(example: dict) -> int:
    return 1 if example["completion"] == "Healthy" else 0


def _myopathy_mapping_function(example: dict) -> int:
    return 1 if example["completion"] == "Myopathy" else 0


def _control_als_myopathy_mapping_function(example: dict) -> int:
    return (
        0
        if example["completion"] == "Healthy"
        else 1 if example["completion"] == "ALS" else 2
    )


def _get_mapping_function(signal_type: str, dataset: str) -> Callable[[dict], int]:
    if signal_type == "EEG":
        if dataset in [
            "trial_1",
            "trial_2",
            "trial_3",
            "trial_4",
            "trial_5",
            "trial_6",
            "trial_7",
            "trial_8",
            "trial_9",
            "trial_10",
        ]:
            return _als_mapping_function

    if signal_type == "EMG":
        if dataset == "control_vs_als_vs_myopathy_dataset":
            return _control_als_myopathy_mapping_function
        if dataset in [
            "control_vs_myopathy_dataset",
            "myopathy_vs_control_and_als_dataset",
        ]:
            return _myopathy_mapping_function
        if dataset == "control_vs_als_and_myopathy_dataset":
            return _control_mapping_function
        if dataset in [
            "als_vs_control_and_myopathy_dataset",
            "als_vs_myopathy_dataset",
            "control_vs_als_dataset",
        ]:
            return _als_mapping_function


def _get_signal_dataset_path(signal_type: str) -> str:
    assert signal_type in get_available_signal_types()

    if signal_type == "EEG":
        return os.path.join(base_path, "eeg_datasets/control_vs_als_dataset")
    if signal_type == "EMG":
        return os.path.join(base_path, "emg_datasets")

    return str()


def get_available_models() -> list:
    return [
        "unsloth/Llama-3.2-1B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "unsloth/Llama-3.1-8B",
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "google/gemma-7b",
        "unsloth/Qwen3-14B",
    ]


def get_available_datasets(signal_type: str) -> list:
    dataset_path = _get_signal_dataset_path(signal_type)
    if dataset_path:
        datasets = os.listdir(dataset_path)
    else:
        datasets = list()

    datasets.sort(key=lambda x: len(x))

    return [d for d in datasets if "." not in d]


def get_available_signal_types() -> list:
    return ["EEG", "EMG"]


def binary_compute_metrics(eval_pred: tuple) -> dict:
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
   
    if logits.shape[1] == 2:
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        probabilities = probs[:, 1]
    else:
        probabilities = logits


    precision = precision_metric.compute(predictions=predictions, references=labels)[
        "precision"
    ]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    roc_auc = roc_auc_score(labels, probabilities)
    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }


def multiclass_compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
        
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")[
        "f1"
    ]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    roc_auc = roc_auc_score(labels, probabilities, multi_class="ovo", average="macro")
    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }


def get_config(
    signal_type: str,
    dataset: str,
    model: str,
    token_len: int = 128,
    lr: float = 1e-4,
    batch_size: int = 16,
    num_epochs: int = 50,
) -> dict:
    signal_types = get_available_signal_types()
    models = get_available_models()
    datasets = get_available_datasets(signal_type)

    assert signal_type in signal_types
    assert dataset in datasets
    assert model in models

    config = dict()

    config["dataset"] = os.path.join(_get_signal_dataset_path(signal_type), dataset)
    config["model"] = model
    config["token_len"] = token_len
    config["lr"] = lr
    config["batch_size"] = batch_size
    config["num_epochs"] = num_epochs
    config["mapping_function"] = _get_mapping_function(signal_type, dataset)
    if signal_type == "EMG" and dataset == "control_vs_als_vs_myopathy_dataset":
        config["num_classes"] = 3
    else:
        config["num_classes"] = 2

    if config["num_classes"] == 2:
        config["compute_metrics"] = binary_compute_metrics
    else:
        config["compute_metrics"] = multiclass_compute_metrics

    return config
