import json
import shutil
import logging
import os
from glob import glob
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def partition_file(input_file, output_directory, chunks=100):
    """Split the input file into smaller chunks."""
    lines = count_lines(input_file)
    logging.info(f"{lines} lines in file")

    chunk_size = round(lines / chunks)
    logging.info(f"Chunk size: {chunk_size} lines")

    curr_lines = []
    count, chunk = 0, 1

    with open(input_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            obj = json.loads(line.strip())
            curr_lines.append(obj)
            count += 1
            if count >= chunk_size:
                save_partition(curr_lines, output_directory, chunk)
                count = 0
                curr_lines = []
                chunk += 1

    # Save the remaining lines, if any.
    if curr_lines:
        save_partition(curr_lines, output_directory, chunk)


def save_partition(json_lines, output_directory, index):
    """Save the current partition of lines to a file."""
    out = os.path.join(output_directory, f"partition-{index}.jsonl")
    logging.info(f"Saving {out}")

    with open(out, 'w', encoding='utf-8') as writer:
        for obj in json_lines:
            writer.write(json.dumps(obj) + '\n')


def count_lines(input_file):
    """Count the number of lines in the file."""
    with open(input_file, 'rb') as f:
        return sum(1 for _ in f)


def join_tagged_files(input_directory, output_file):
    """Join all tagged files from the input directory into one output file."""
    tagged_files = glob(os.path.join(input_directory, "*.jsonl"))

    logging.info(f"Found {len(tagged_files)} files to merge")

    with open(output_file, 'w', encoding='utf-8') as writer:
        for tagged_file in tqdm(tagged_files, desc="Merging tagged files"):
            with open(tagged_file, 'r', encoding='utf-8') as reader:
                for line in reader:
                    writer.write(line)


def delete_partitioned_files(dir_path):
    """Delete all partitioned files in the directory."""
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        logging.error(f"Error deleting directory {dir_path}: {e.strerror}")

# Example usage:
# partition_file('/path/to/large_input.jsonl', '/path/to/output_directory', chunks=100)
# join_tagged_files('/path/to/output_directory', '/path/to/joined_output.jsonl')
# delete_partitioned_files('/path/to/output_directory')
