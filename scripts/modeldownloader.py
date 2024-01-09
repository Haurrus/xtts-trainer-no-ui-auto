import requests
from tqdm import tqdm

def create_directory_if_not_exists(directory):
    if not directory.exists():
        directory.mkdir(parents=True)

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

def download_model(this_dir,model_version):
    # Define paths
    base_path = this_dir / 'models'
    model_path = base_path / f'{model_version}'

    # Define files and their corresponding URLs
    files_to_download = {
         "config.json": f"https://huggingface.co/coqui/XTTS-v2/raw/{model_version}/config.json",
         "model.pth": f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_version}/model.pth?download=true",
         "vocab.json": f"https://huggingface.co/coqui/XTTS-v2/raw/{model_version}/vocab.json",
         "speakers_xtts.pth": f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_version}/speakers_xtts.pth?download=true"
    }

    # Check and create directories
    create_directory_if_not_exists(base_path)
    create_directory_if_not_exists(model_path)

    # Download files if they don't exist
    for filename, url in files_to_download.items():
         destination = model_path / filename
         if not destination.exists():
             print(f"[XTTS] Downloading {filename}...")
             download_file(url, destination)

