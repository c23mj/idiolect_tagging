import os
from glob import glob
from multiprocessing import Pool
import json
import torch
import time

# Global variables for tokenizer and model
tokenizer = None
model = None

def initialize_worker():
    global tokenizer, model
    from tokenization_enc_t5 import EncT5Tokenizer
    from modeling_enc_t5 import EncT5ForSequenceClassification
    tokenizer = EncT5Tokenizer.from_pretrained("t5-base")
    model = EncT5ForSequenceClassification.from_pretrained(
        "/shared/3/projects/hiatus/idiolect/models/stylegenome_lisa_sfam/lisa_checkpoint",
        num_labels=768, problem_type="regression"
    )
    model.eval()
    print(f"Worker {os.getpid()} initialized")

def tag_lisa(obj):
    global tokenizer, model
    start_time = time.time()  # Start timing
    try:
        tokenized = tokenizer(
            [obj['body']],
            truncation=True, max_length=512, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            prediction = model.forward(**tokenized)[0][0].cpu().float()
        vector = torch.clamp(prediction, min=0.0, max=1.0).tolist()
        obj['lisa_vector'] = vector
        print(f"Tagged file {obj['id']}")
    except Exception as e:
        print(f"Error processing object {obj['id']}: {e}")
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Time taken to tag document {obj['id']}: {elapsed_time:.4f} seconds")  # Print the elapsed time
    return obj

def tag_partition(input_file, output_file):
    print(f"Tagging file {input_file} with worker {os.getpid()}")
    tagged_objects = []
    try:
        with open(input_file, 'r') as reader:
            for line in reader:
                obj = json.loads(line.strip())
                tagged_object = tag_lisa(obj)
                tagged_objects.append(tagged_object)
                
                # Process in batches of 50 to reduce overhead
                if len(tagged_objects) % 50 == 0:
                    print(f"Appending chunk of 50 objects to {output_file}")
                    append_chunk(output_file, tagged_objects)
                    tagged_objects = []

        # Write any remaining objects that didn't make up a full batch
        if tagged_objects:
            append_chunk(output_file, tagged_objects)
            print(f"Appending final chunk to {output_file}")
    except Exception as e:
        print(f"Error tagging partition {input_file}: {e}")

def build_process_args(input_directory, output_directory):
    partition_files = glob(f"{input_directory}/*.jsonl")
    process_args = []

    for fp in partition_files:
        fname = os.path.basename(fp).replace('.jsonl', '') + '-tagged.jsonl'
        out = os.path.join(output_directory, fname)
        process_args.append((fp, out))

    return process_args

def append_chunk(output_file, tagged_objects):
    try:
        with open(output_file, 'a') as writer:
            for obj in tagged_objects:
                writer.write(json.dumps(obj) + '\n')
    except Exception as e:
        print(f"Error appending chunk to {output_file}: {e}")

def tag_partitions(input_directory, output_directory, num_workers):
    process_args = build_process_args(input_directory, output_directory)

    # Ensure we never use more workers than specified
    num_workers = min(num_workers, len(process_args))

    print(f"Using {num_workers} workers for tagging")
    with Pool(num_workers, initializer=initialize_worker) as pool:
        pool.starmap(tag_partition, process_args)
    print("Finished multiprocessing pool")

# Example usage:
# tag_partitions('/path/to/input', '/path/to/output', 2)
