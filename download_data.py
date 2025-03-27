import os
import gdown

dataset_url = "https://drive.google.com/uc?id=1DruMwtT27c0azMzKcMNlfBVj44sjrZgL"
dataset_path = "datasets/Dataset_Specific_Unlabelled.h5"

os.makedirs("datasets", exist_ok=True)

if not os.path.exists(dataset_path):
    print(f"Downloading dataset to {dataset_path}...")
    gdown.download(dataset_url, dataset_path, quiet=False)
    print("Download complete.")
else:
    print("Dataset already exists. Skipping download.")
