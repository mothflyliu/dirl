import os
import pickle

def find(file_name):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    directories_to_search = [current_dir, parent_dir]
    for directory in directories_to_search:
        for root, dirs, files in os.walk(directory):
            if file_name in files:
                return os.path.join(root, file_name)
    return None
def found_file_path(demo_file_name):
    if demo_file_name:
        found_file_path = find(demo_file_name)
        if found_file_path:
            a = found_file_path
        else:
            print(f"Could not find file: {demo_file_name}")
    else:
        print("No 'demo_file' specified in job data.")
    return a