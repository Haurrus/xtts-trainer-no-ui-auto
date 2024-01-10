# Preface

## Prerequisites
- Python 3.11.x
- CUDA-enabled device 11.8

## Installation
Follow these steps for installation:

1. Ensure that `CUDA` is installed
2. Clone the repository: `git clone https://github.com/Haurrus/xtts-trainer-no-ui-auto`
3. Navigate into the directory: `cd xtts-trainer-no-ui-auto`
4. Create a virtual environment: `python -m venv venv`
5. Activate the virtual environment:
   - On Windows use : `venv\scripts\activate`
   - On linux use    : `source venv\bin\activate`

6. Install PyTorch and torchaudio with pip command :

   `pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118`

7. Install all dependencies from requirements.txt :

    `pip install -r requirements.txt`

# XTTS Fine-Tuning Project for xTTSv2 xtts_finetune_no_ui_auto.py

## Overview
This is a Python script for fine-tuning a text-to-speech (TTS) model for xTTSv2. The script utilizes custom datasets and use CUDA for accelerated training.

## Usage
To use the script, you need to specify two JSON files: `args.json` and `datasets.json`.

### args.json
This file should contain the following key parameters:
- `num_epochs`: Number of epochs for training, if set to 0 it will auto calculate it.
- `batch_size`: Batch size for training.
- `grad_acumm`: Gradient accumulation steps.
-  `max_audio_length`: max audio duration of wavs used to train.
-  `language`: language used to train the model.
-  `version`: by default main from xTTSv2
-  `json_file`: by default main from xTTSv2
-  `custom_model`: by default main from xTTSv2

### datasets.json
This file should list the datasets to be used with paths and activation flags.

### Finetune_models folder
To train models you need a dataset, there's an exemple dataset in the finetune_models, it's a FemaleDarkElf voice from Skyrim

### Running the Script
Execute the script with the following command:
```
python xtts_finetune_noweb_auto.py --args_json path/to/args.json --datasets_json path/to/datasets.json
```

## Features
- Custom model training and fine-tuning.
- Support for multiple datasets.


# Audio Dataset Preprocessing xtts_generate_dataset.py

## Overview
This script processes audio files to create training and evaluation datasets using the Whisper model. It has been updated to include several new features and improvements.

## Usage

To use the script, provide the path to a JSON configuration file and the Whisper model version as command-line arguments:

```
python xtts_generate_dataset.py --config path/to/config.json --whisper_version large-v3
```

The JSON configuration file should contain the audio paths, target language, activation flag, and name for each dataset.

## JSON Configuration Format

The configuration file should follow this format:

```json
[
    {
	"name": "dataset_name"
        "audio_path": "path/to/audio/files",
        "language": "en",
        "activate": true,
    },
]
```

Replace `path/to/audio/files` with the actual path to your audio files and `dataset_name` with a preferred name for your output subdirectory.

## Features

1. **Processing Entire Audio Files**: The script has been modified to process entire audio files without splitting them into segments. Each audio file is transcribed as a whole, and the corresponding transcription is stored.

2. **Output Directory Customization**: The output directory is now named `output_datasets` and is created in the root directory where the script is executed. Inside this directory, subdirectories are created based on the `name` provided in the JSON configuration file.

3. **Language Configuration**: The script writes the target language to a `lang.txt` file in the output directory, ensuring consistent language settings across the dataset.

4. **Audio File Copying**: All processed audio files are copied into a `wavs` folder located in their respective output subdirectories.

5. **Error Handling and Logging**: The script includes error handling and logging mechanisms to provide clear feedback in case of any issues during the processing.

6. **Configurable Through JSON**: The entire preprocessing can be configured using a JSON file, making it easy to adjust settings like the target language, audio paths, and output names.


## Contributing
Contributions are welcome. Please fork the repository and submit pull requests with your changes.


## Credit
Thanks to the author **daswer123** for the repository [xtts-webui](https://github.com/daswer123/xtts-webui) , My project is based on his work.
