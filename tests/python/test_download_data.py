from pathlib import Path
import requests, zipfile

def download(url: str, filename: str) -> None:
    filename += '.zip'
    PATH = Path(__file__).with_name('data')
    PATH.mkdir(parents=True, exist_ok=True)

    # Download the data if it doesn't exist
    if not (PATH / filename).exists():
        content = requests.get(url).content
        (PATH / filename).open("wb").write(content)

    with zipfile.ZipFile(PATH/filename, 'r') as zip_ref:
        print(PATH/filename)
        zip_ref.extractall(PATH)