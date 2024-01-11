import argparse
import json
import os
from pathlib import Path
import torch
import traceback
import shutil
from scripts.utils.gpt_train import train_gpt
import math
import csv
from scripts.modeldownloader import download_model

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def calculate_epochs(total_samples, batch_size, target_steps):
    """
    Calculate the number of epochs based on the target number of steps.

    :param total_samples: Total number of training samples in the dataset.
    :param batch_size: Number of samples processed in one iteration.
    :param target_steps: Target number of steps to reach.
    :return: Number of epochs needed to reach the target steps.
    """
    steps_per_epoch = total_samples / batch_size
    num_epochs = math.ceil(target_steps / steps_per_epoch)
    return num_epochs

def count_samples_in_csv(csv_file_path):
    """
    Count the number of samples in a CSV file. Assumes each line is a sample.

    :param csv_file_path: Path to the CSV file.
    :return: Number of samples in the CSV file.
    """
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        sample_count = sum(1 for row in reader)
    return sample_count

def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    clear_gpu_cache()
    run_dir = Path(output_path) / "run"
    if run_dir.exists():
        os.remove(run_dir)

    if not train_csv or not eval_csv:
        raise Exception("Missing `Train CSV` or `Eval CSV` fields")

    try:
        max_audio_length = int(max_audio_length * 22050)
        # Use custom_model only if it's not None
        if custom_model and not os.path.exists(custom_model):
            raise Exception(f"Custom model path does not exist: {custom_model}")
        elif not custom_model:
            custom_model = None  # Explicitly setting to None if it's not provided

        speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
            custom_model, version, language, num_epochs, batch_size, grad_acumm, 
            train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length
        )
    except Exception as e:
        traceback.print_exc()
        raise e

    ready_dir = Path(output_path) / "ready"
    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
    ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")
    speaker_reference_path = Path(speaker_wav)
    speaker_reference_new_path = ready_dir / "reference.wav"
    shutil.copy(speaker_reference_path, speaker_reference_new_path)
    print("Model training done!")
    return config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, speaker_reference_new_path

def optimize_model(out_path, clear_train_data):
    out_path = Path(out_path)
    ready_dir = out_path / "ready"
    run_dir = out_path / "run"
    dataset_dir = out_path / "dataset"
    if clear_train_data in {"run", "all"} and run_dir.exists():
        shutil.rmtree(run_dir)
    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    model_path = ready_dir / "unoptimize_model.pth"
    if not model_path.is_file():
        raise Exception("Unoptimized model not found in ready folder")
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]
    os.remove(model_path)
    optimized_model_file_name = "model.pth"
    optimized_model = ready_dir / optimized_model_file_name
    torch.save(checkpoint, optimized_model)
    ft_xtts_checkpoint = str(optimized_model)
    clear_gpu_cache()
    return ft_xtts_checkpoint

def load_params(out_path):
    print(f"Checking dataset in path: {out_path}")  # Diagnostic print
    dataset_path = Path(out_path)
    print(f"Looking for dataset at: {dataset_path}")  # Diagnostic print
    if not dataset_path.exists():
        raise Exception(f"Output folder does not exist at {dataset_path}")
    eval_train = dataset_path / "metadata_train.csv"
    eval_csv = dataset_path / "metadata_eval.csv"
    lang_file_path = dataset_path / "lang.txt"
    if not lang_file_path.exists():
        raise Exception(f"lang.txt not found at {lang_file_path}")
    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
        current_language = existing_lang_file.read().strip()
    clear_gpu_cache()
    return eval_train, eval_csv, current_language



def read_dataset_queue(json_file):
    with open(json_file, 'r') as file:
        datasets = json.load(file)
    active_datasets = [d['path'] for d in datasets if d.get('activate', False)]
    return active_datasets

def load_arguments_from_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--args_json", type=str, help="Path to the JSON file containing additional arguments")
    parser.add_argument("--datasets_json", type=str, help="Path to the JSON file containing dataset queue")
    args = parser.parse_args()
        # Check if the necessary arguments are not provided
    if not args.args_json or not args.datasets_json:
        parser.print_help()
        exit(1)
    # Load additional arguments from JSON file
    if args.args_json:
        json_args = load_arguments_from_json(args.args_json)
    else:
        json_args = {}

    base_dir = Path(__file__).parent
    # Read dataset queue from datasets.json
    dataset_queue = read_dataset_queue(args.datasets_json) if args.datasets_json else []
    for dataset_path_str in dataset_queue:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.is_absolute():
            dataset_path = base_dir / dataset_path
        try:
            train_csv, eval_csv, current_language = load_params(dataset_path)
            # Accessing the arguments
            version = json_args.get("version", "main") or "main"
            num_epochs = json_args.get("num_epochs", 0)
            batch_size = json_args.get("batch_size", 2)
            grad_acumm = json_args.get("grad_acumm", 1)
            max_audio_length = json_args.get("max_audio_length", 11)
            language = json_args.get("language", "fr")
            
            # Determine the path for custom model
            custom_model = json_args.get("custom_model")
            if not custom_model:
                this_dir = Path(__file__).parent
                download_model(this_dir, version)  # Ensure this function is defined in modeldownloader.py
                custom_model = this_dir / f'models/{version}/model.pth'
            else:
                custom_model = Path(custom_model)
            custom_model = str(custom_model)
            
            # Determine the number of epochs
            if num_epochs == 0:
                total_samples = count_samples_in_csv(train_csv)
                print(f"Number of total_samples : {total_samples}")
                if total_samples > 1000:
                    target_steps = 30000
                elif 600 <= total_samples < 1000:
                    target_steps = 20000
                elif 250 <= total_samples < 600:
                    target_steps = 15000
                else:
                    target_steps = 10000
                num_epochs = calculate_epochs(total_samples, batch_size, target_steps)
                print(f"Number of target_steps : {target_steps}")
                print(f"Number of epochs needed: {num_epochs}")
            config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, speaker_reference_new_path = train_model(
                custom_model, version, current_language, train_csv, eval_csv, 
                num_epochs, batch_size, grad_acumm, dataset_path, max_audio_length
            )
            optimized_model_path = optimize_model(dataset_path, None)
            # Additional steps to load and use the model can be implemented here
        except Exception as e:
            print(f"Error processing dataset {dataset_path}: {e}")

