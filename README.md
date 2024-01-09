# XTTS Fine-Tuning Project

## Overview
This project includes a Python script for fine-tuning a text-to-speech (TTS) model. The script utilizes custom datasets and use CUDA for accelerated training.

## Prerequisites
- Python 3.11.7
- CUDA-enabled device 11.8

## Installation
1. Clone the repository:
   ```
   git clone [Your Repository URL]
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   If you are using a CUDA-enabled device:
   ```
   pip install -r requirements_cuda.txt
   ```

## Usage
To use the script, you need to specify two JSON files: `args.json` and `datasets.json`.

### args.json
This file should contain the following key parameters:
- `num_epochs`: Number of epochs for training, if set to 0 it will auto calculate it.
- `batch_size`: Batch size for training.
- `grad_acumm`: Gradient accumulation steps.
-  `max_audio_length`: max audio duration of wavs used to train.
-  `language`: language used to train the model.
-  `version`: "" by default main from xTTSv2
-  `json_file`: "" by default main from xTTSv2
-  `custom_model`: "" by default main from xTTSv2

### datasets.json
This file should list the datasets to be used with paths and activation flags.

### Running the Script
Execute the script with the following command:
```
python xtts_finetune_noweb_auto.py --args_json path/to/args.json --datasets_json path/to/datasets.json
```

## Features
- Custom model training and fine-tuning.
- Support for multiple datasets.

## Contributing
Contributions are welcome. Please fork the repository and submit pull requests with your changes.


## Credit
Thanks to the author **daswer123 ** for the repository [xtts-webui](https://github.com/daswer123/xtts-webui) , My project is based on his work.
