import os
import json
import argparse
import traceback
from scripts.utils.formatter import format_audio_list

def preprocess_audio_dataset(audio_dir, target_language, whisper_version, out_path, name):
    # Create a subdirectory within out_path using the 'name' parameter
    named_out_path = os.path.join(out_path, name)

    # Ensure the named output path exists
    os.makedirs(named_out_path, exist_ok=True)

    # List audio files in the directory
    audio_paths = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]

    # Process the audio files
    try:
        train_meta, eval_meta, audio_total_size = format_audio_list(
            audio_paths, 
            whisper_model=whisper_version, 
            target_language=target_language, 
            out_path=named_out_path,
            speaker_name=name
        )
        print("ok3")
        # Check if total audio length is sufficient
        if audio_total_size < 120:
            return "The sum of the duration of the audios should be at least 2 minutes!"

        print("Dataset Processed!")
        return "Dataset Processed!", train_meta, eval_meta

    except Exception as e:
        traceback.print_exc()
        return f"The data processing was interrupted due to an error: {e}"

def load_dataset_config(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets from audio files.")
    parser.add_argument("--config", type=str, required=True, help="Path to the dataset configuration JSON file.")
    parser.add_argument("--whisper_version", type=str, required=True, help="Whisper model version can be .")
    args = parser.parse_args()

    dataset_configs = load_dataset_config(args.config)
    print(dataset_configs)

    # Set the base output path to 'output_datasets' in the current directory
    base_out_path = os.path.join(os.getcwd(), "output_datasets")
    os.makedirs(base_out_path, exist_ok=True)

    for config in dataset_configs:
        if config.get("activate", True):
            audio_dir = config["audio_path"]
            language = config["language"]
            name = config.get("name", "default_name") 
            preprocess_audio_dataset(audio_dir, language, args.whisper_version, base_out_path, name)
