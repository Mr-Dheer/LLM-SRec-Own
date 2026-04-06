"""
Utility functions for LLM-SRec.
Provides basic file system helpers used across the project.
"""

import os


def create_dir(directory):
    """Create a directory if it does not already exist (used for saving model checkpoints)."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def find_filepath(target_path, target_word):
    """
    Find all files in `target_path` whose filenames contain `target_word`.

    Used primarily by recsys_model.py to locate the pre-trained SASRec checkpoint
    file (e.g., finding the single .pth file in the SASRec model directory).

    Args:
        target_path: Directory to search in.
        target_word: Substring to match in filenames (e.g., '.pth', '.csv').

    Returns:
        List of full file paths matching the criteria.
    """
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)

    return file_paths
