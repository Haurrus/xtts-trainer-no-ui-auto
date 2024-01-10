import os
import torchaudio
import shutil
import pandas as pd
from faster_whisper import WhisperModel
from tqdm import tqdm
from scripts.utils.tokenizer import multilingual_cleaners
import torch

torch.set_num_threads(16)

audio_types = (".wav", ".mp3", ".flac")


def format_audio_list(audio_files, target_language="en", whisper_model="large-v3", out_path=None,speaker_name="coqui", eval_percentage=0.15):
    # Ensure that output directory and wavs subdirectory exist
    os.makedirs(out_path, exist_ok=True)
    wavs_path = os.path.join(out_path, "wavs")
    os.makedirs(wavs_path, exist_ok=True)

    # Write the target language to lang.txt in the output directory
    lang_file_path = os.path.join(out_path, "lang.txt")
    with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
        lang_file.write(target_language + '\n')
    
    # Load Whisper Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = WhisperModel(whisper_model, device=device, compute_type="float16")

    # Initialize metadata
    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    audio_total_size = 0

    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        # Load audio file
        wav, sr = torchaudio.load(audio_path)
        
        # Convert stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Update total audio size
        audio_total_size += wav.size(-1) / sr

        # Transcribe audio
        transcription_segments, _ = asr_model.transcribe(audio_path, language=target_language)
        
        # Extract and concatenate the transcription text from the generator
        full_transcription = ' '.join(segment.text for segment in transcription_segments)

        # Save metadata
        audio_file_name = os.path.basename(audio_path)
        metadata["audio_file"].append(audio_file_name)
        metadata["text"].append(full_transcription)
        metadata["speaker_name"].append(speaker_name)

        # Copy audio file to wavs folder
        destination_path = os.path.join(wavs_path, audio_file_name)
        shutil.copy2(audio_path, destination_path)

    # Convert metadata to DataFrame
    metadata_df = pd.DataFrame(metadata)

    # Split into training and evaluation sets
    train_df = metadata_df.sample(frac=1 - eval_percentage).reset_index(drop=True)
    eval_df = metadata_df.drop(train_df.index).reset_index(drop=True)

    # Save to CSV
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")

    train_df.to_csv(train_metadata_path, sep='|', index=False)
    eval_df.to_csv(eval_metadata_path, sep='|', index=False)


    # Clean up resources
    del asr_model

    return train_metadata_path, eval_metadata_path, audio_total_size, lang_file_path
