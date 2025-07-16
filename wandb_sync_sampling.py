import wandb
import os, time
import subprocess
from tqdm import tqdm


def is_file_updated_inwandb(directory, extension=".wandb", minutes=60 * 1):
    # Get the current time in seconds since the epoch
    current_time = time.time()

    # Get the time 10 minutes ago in seconds since the epoch
    ten_minutes_ago = current_time - (minutes * 60)

    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(foldername, filename)
                # Get the time the file was last modified
                file_mtime = os.path.getmtime(file_path)
                # If the file was modified in the last 10 minutes, return True
                if file_mtime > ten_minutes_ago:
                    return True

    # If no files were modified in the last 10 minutes, return False
    return False


if __name__ == "__main__":
    root = "samples"
    command_list = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        if "wandb" in dirnames:
            wandb_dir = os.path.join(dirpath, "wandb")
            latest_run_dir = os.path.join(wandb_dir, "latest-run")
            if os.path.exists(latest_run_dir):
                print(f"Found latest-run directory: {latest_run_dir}")
                if is_file_updated_inwandb(latest_run_dir):
                    command_list.append(
                        f"wandb sync {latest_run_dir} --append"
                    )

    for command in tqdm(command_list, total=len(command_list)):
        print(command)
        try:
            subprocess.run(command, shell=True)
        except Exception as e:
            print(e)
            continue
        print("********** sync done **********")
