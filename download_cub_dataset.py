#!/usr/bin/env python3
"""
Script to download and extract the CUB-200-2011 dataset.
Downloads from the official source and organizes into the expected directory structure.
"""

import os
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm
import shutil


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    # Create the target directory
    target_dir = "data/CUB_200_2011"
    os.makedirs(target_dir, exist_ok=True)
    
    # CUB-200-2011 dataset URLs
    base_url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011"
    files = {
        "images": f"{base_url}/CUB_200_2011.tgz",
        "attributes": f"{base_url}/attributes.txt",
        "bounding_boxes": f"{base_url}/bounding_boxes.txt",
        "classes": f"{base_url}/classes.txt",
        "image_class_labels": f"{base_url}/image_class_labels.txt",
        "images_txt": f"{base_url}/images.txt",
        "train_test_split": f"{base_url}/train_test_split.txt"
    }
    
    print("Downloading CUB-200-2011 dataset...")
    
    # Download the main dataset file
    main_file = os.path.join(target_dir, "CUB_200_2011.tgz")
    if not os.path.exists(main_file):
        print(f"Downloading main dataset file...")
        download_url(files["images"], main_file)
    else:
        print("Main dataset file already exists, skipping download.")
    
    # Download additional metadata files
    for filename, url in files.items():
        if filename == "images":
            continue  # Already downloaded above
        
        filepath = os.path.join(target_dir, f"{filename}.txt")
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            download_url(url, filepath)
        else:
            print(f"{filename} already exists, skipping download.")
    
    # Extract the main dataset
    print("Extracting dataset...")
    extract_dir = os.path.join(target_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    if not os.path.exists(os.path.join(extract_dir, "CUB_200_2011")):
        with tarfile.open(main_file, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print("Dataset extracted successfully.")
    else:
        print("Dataset already extracted, skipping.")
    
    # Move images to the expected location
    expected_images_dir = os.path.join(target_dir, "images")
    extracted_images_dir = os.path.join(extract_dir, "CUB_200_2011", "images")
    
    if not os.path.exists(expected_images_dir):
        print("Moving images to expected location...")
        shutil.move(extracted_images_dir, expected_images_dir)
        print("Images moved successfully.")
    else:
        print("Images directory already exists.")
    
    # Clean up extracted directory
    if os.path.exists(extract_dir):
        print("Cleaning up temporary files...")
        shutil.rmtree(extract_dir)
    
    # Verify the structure
    if os.path.exists(expected_images_dir):
        num_classes = len([d for d in os.listdir(expected_images_dir) 
                          if os.path.isdir(os.path.join(expected_images_dir, d))])
        print(f"Dataset successfully downloaded and organized!")
        print(f"Found {num_classes} classes in {expected_images_dir}")
        print(f"Dataset is ready to use at: {target_dir}")
    else:
        print("Error: Images directory not found after setup.")


if __name__ == "__main__":
    main() 