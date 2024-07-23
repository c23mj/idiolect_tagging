import os
from glob import glob
from multiprocessing import Pool
import json
import torch

def initialize_worker():
    global tokenizer, model
    from tokenization_enc_t5 import EncT5Tokenizer
    from modeling_enc_t5 import EncT5ForSequenceClassification

    tokenizer = EncT5Tokenizer.from_pretrained("t5-base")
    model = EncT5ForSequenceClassification.from_pretrained(
        "/shared/3/projects/hiatus/idiolect/models/stylegenome_lisa_sfam/lisa_checkpoint",
        num_labels=768, problem_type="regression"
    )
    print(f"Worker {os.getpid()} initialized")

def tag_partitions(input_directory, output_directory, num_workers):
    process_args = build_process_args(input_directory, output_directory)
    
    os.sched_setaffinity(0, range(num_workers))
    print(f"Setting CPU affinity to use {num_workers} CPUs")

    print("Starting multiprocessing pool...")
    with Pool(num_workers, initializer=initialize_worker, initargs=()) as pool:
        pool.starmap(tag_partition, process_args)
    print("Finished multiprocessing pool")

def tag_lisa(obj):
    global tokenizer, model
    try:
        tokenized = tokenizer(
            [obj['fullText']],
            truncation=True, max_length=512, padding=True, return_tensors="pt"
        )
        print(f"Tokenized input for object: {obj['documentID']}")
        prediction = model.forward(**tokenized)[0][0].cpu().detach().float()
        print(f"Prediction for object {obj['documentID']}: {prediction}")
        vector = torch.clamp(prediction, min=0.0, max=1.0).numpy().tolist()
        if 'encodings' in obj:
            del obj['encodings']
        obj['lisa_vector'] = vector
    except Exception as e:
        print(f"Error processing object {obj['documentID']}: {e}")
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
                print(f"Tagged object {obj['documentID']} for worker {os.getpid()}. Count: {len(tagged_objects)}")

                if len(tagged_objects) % 10 == 0:
                    print(f"Appending chunk of 10 objects to {output_file}")
                    append_chunk(output_file, tagged_objects)
                    tagged_objects = []

        if tagged_objects:
            append_chunk(output_file, tagged_objects)
    except Exception as e:
        print(f"Error tagging partition {input_file}: {e}")

def build_process_args(input_directory, output_directory):
    partition_files = glob(f"{input_directory}/*.jsonl")
    process_args = []

    for fp in partition_files:
        fname = fp.rsplit('/', 1)[-1].replace('.jsonl', '') + '-tagged.jsonl'
        out = f"{output_directory}/{fname}"
        process_args.append((fp, out))

    return process_args

def append_chunk(output_file, tagged_objects):
    try:
        with open(output_file, 'a') as writer:
            for obj in tagged_objects:
                writer.write(json.dumps(obj) + '\n')
        print(f"Written {len(tagged_objects)} objects to {output_file}")
    except Exception as e:
        print(f"Error appending chunk to {output_file}: {e}")

# Example usage:
# tag_partitions('/path/to/input', '/path/to/output', 4)