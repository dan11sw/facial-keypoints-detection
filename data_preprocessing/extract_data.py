import zipfile
import os

def extract_files(filepath="./data"):
    if not os.path.exists(filepath + "/training.csv"):
        with zipfile.ZipFile(filepath + "/training.zip") as zip_ref:
            zip_ref.extractall(filepath)

    if not os.path.exists(filepath + "/test.csv"):
        with zipfile.ZipFile(filepath + "/test.zip") as zip_ref:
            zip_ref.extractall(filepath)