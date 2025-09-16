# ALS-EMG

This project explores how machine learning and large language models can help detect and classify ALS (Amyotrophic Lateral Sclerosis) and Myopathy from EMG and EEG signals.

## Requirements

- Python 3.10+ (specify your Python version here)
- Virtual environment support
- GPU resources recommended for LLM fine-tuning (optional but preferred)
- HuggingFace API key
- WandB API key

## Setup Instructions

### 1. Clone the Repository
First, clone this repository to your local machine:

```bash
git clone https://github.com/ChatMED/ALS-EMG.git
```

### 2. Environment Setup

Navigate to the project folder and create a virtual environment:

```bash
cd ALS-EMG
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. EMG - Evaluation and Feature Generation

Navigate to `EMG` and run the evaluation:

```bash
cd EMG
jupyter notebook DatasetXGBoostEvaluation.ipynb
```

#### Optional: Feature Generation for EMG
If you want to regenerate features:

1. Download `EMGLAB2001` from [here](https://www.kaggle.com/datasets/lydialoubar/emglab)
2. Place the `emglab` folder into `EMG/als_datasets`
3. Rename the `emglab` folder to `emglab_dataset_bb`
4. Run the feature extraction script:
   ```bash
   python emg_feature_extraction.py
   ```

**⚠️ Note:** Running `emg_feature_extraction.py` will overwrite the current datasets in `EMG`.

**⚠️ Note:** Running the code in `DatasetPreparation.ipynb` will overwrite the datasets in `LLM` for LLM fine-tuning for `EMG`.

### 4. EEG - Evaluation and Feature Generation

Navigate to `EEG` and run the evaluation:

```bash
cd EEG
jupyter notebook DatasetXGBoostEvaluation.ipynb
```

#### Optional: Feature Generation for EEG
If you want to regenerate features:

1. Download `EEGET-ALS` from [here](https://www.kaggle.com/datasets/patrickiitmz/eeget-als-dataset)
2. Place the `EEGET-ALS` folder into `EEG`
2. Rename the `EEGET-ALS` folder to `EEGET-ALS Dataset`
3. Run the feature extraction script:
   ```bash
   python eeg_feature_extraction.py
   ```

**⚠️ Note:** Running `eeg_feature_extraction.py` will overwrite the current datasets in `EEG`.

**⚠️ Note:** Running the code in `DatasetPreparation.ipynb` will overwrite the datasets in `LLM` for LLM fine-tuning for `EEG`.

### 5. LLM - LLM Fine-tuning

Navigate to `LLM`:

```bash
cd LLM
```

#### Environment Configuration
Create a `.env` file in `LLM` with your API keys:

```bash
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
```

#### Running Experiments
Execute all experiments:

```bash
python running_script.py
```

The `results` folder and its contents will be created after execution.

#### Resource Optimization
If you encounter GPU resource limitations:

- Run a smaller number of experiments (follow instructions in `running_script.py`)
- Use a smaller model:
  - Check available models in `config.py`
  - Modify the model selection in `model_finetuning_script.py`

#### Google Colab Setup
If running in Google Colab, update the `base_path` variable to point to your project directory path inside Google Drive.

## Project Structure

```
project-folder/
├── requirements.txt
├── venv/
├── EMG/
│   ├── DatasetXGBoostEvaluation.ipynb
│   ├── emg_feature_extraction.py
│   ├── DatasetPreparation.ipynb
│   └── als_datasets/
│       ├── emglab_dataset_bb/
│       └── ...
├── EEG/
│   ├── DatasetXGBoostEvaluation.ipynb
│   ├── eeg_feature_extraction.py
│   ├── DatasetPreparation.ipynb
│   ├── classification_datasets/
│   └── EEGET-ALS Dataset/
│
│
├── LLM/
│   ├── .env
│   ├── running_script.py
│   ├── config.py
│   ├── model_finetuning_script.py
│   └── results/ (created after execution)

```

## Important Notes

- Always activate your virtual environment before running any scripts
- Be cautious when running feature extraction scripts as they overwrite existing datasets
- Ensure you have sufficient GPU resources for LLM fine-tuning
- Keep your API keys secure

## Troubleshooting

- If you encounter memory issues during fine-tuning, try using a smaller model or reducing batch size
- For dataset preparation issues, ensure all required datasets are properly downloaded and placed in the correct directories