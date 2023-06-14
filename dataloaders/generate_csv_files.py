from utils import *
from global_vars import *
import numpy as np
import random
import os

DATA_SPLIT = 0.95   # percentage of data to be used for training
CSV_PATH = 'dataloaders/dataset_files'

def save_file(fname, dict_name):
    npa = np.array([dict_name['subject'], dict_name['sequence'], dict_name['num_frames']]).T
    np.savetxt(os.path.join(CSV_PATH, fname), npa, delimiter=',', fmt='%s')
    print(f"Successfully created {fname} with {npa.shape[0] - 1} entries")

print(f"Your Dataset Location is set to '{DEX_YCB_DIR}'")
print(f"Creating csv files in {CSV_PATH} with train/validate ratio of {DATA_SPLIT}")

train_dict_summary = {'subject': ['subject'], 'sequence': ['sequence'], 'num_frames': ['num_frames']}
validate_dict_summary = {'subject': ['subject'], 'sequence': ['sequence'], 'num_frames': ['num_frames']}
train_dict = {'subject': ['subject'], 'sequence': ['sequence'], 'num_frames': ['num_frames']}
validate_dict = {'subject': ['subject'], 'sequence': ['sequence'], 'num_frames': ['num_frames']}
os.makedirs(CSV_PATH, exist_ok=True)

for subject in SUBJECTS:
    subject_path = os.path.join(DEX_YCB_DIR, subject)
    sequences = os.listdir(subject_path)
    random.shuffle(sequences)
    total_seqs = len(sequences)
    split_idx = int(DATA_SPLIT * total_seqs)
    train_sequences = sequences[:split_idx]
    validate_sequences = sequences[split_idx:]
    print(f" Working on {subject} ...", end = '\r')
    for sequence in train_sequences:
        sequence_path = os.path.join(subject_path, sequence)
        meta_file_path = os.path.join(sequence_path, 'meta.yml')
        meta_file = load_yml(meta_file_path)
        num_frames = meta_file['num_frames']
        train_dict_summary['subject'].append(subject)
        train_dict_summary['sequence'].append(sequence)
        train_dict_summary['num_frames'].append(num_frames)
        for frame in range(2, num_frames-2):
            train_dict['subject'].append(subject)
            train_dict['sequence'].append(sequence)
            train_dict['num_frames'].append(frame)

    for sequence in validate_sequences:
        sequence_path = os.path.join(subject_path, sequence)
        meta_file_path = os.path.join(sequence_path, 'meta.yml')
        meta_file = load_yml(meta_file_path)
        num_frames = meta_file['num_frames']
        validate_dict_summary['subject'].append(subject)
        validate_dict_summary['sequence'].append(sequence)
        validate_dict_summary['num_frames'].append(num_frames)
        for frame in range(2, num_frames-2):
            validate_dict['subject'].append(subject)
            validate_dict['sequence'].append(sequence)
            validate_dict['num_frames'].append(frame)

save_file('train_dataset.csv', train_dict)
save_file('validate_dataset.csv', validate_dict)
save_file('train_dataset_summary.csv', train_dict_summary)
save_file('validate_dataset_summary.csv', validate_dict_summary)
