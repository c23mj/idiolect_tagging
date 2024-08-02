import os
from glob import glob
from multiprocessing import Pool, cpu_count
import json
import torch
# from time import time

def initialize_worker():
    global tokenizer, model
    from tokenization_enc_t5 import EncT5Tokenizer
    from modeling_enc_t5 import EncT5ForSequenceClassification
    tokenizer = EncT5Tokenizer.from_pretrained("t5-base")
    model = EncT5ForSequenceClassification.from_pretrained(
        "/shared/3/projects/hiatus/idiolect/models/stylegenome_lisa_sfam/lisa_checkpoint",
        num_labels=768, problem_type="regression"
    )

def tag_lisa(obj):
    global tokenizer, model
#     start_time = time()  # Start timing
    try:
        tokenized = tokenizer(
            [obj['fullText']],
            truncation=True, max_length=512, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            prediction = model.forward(**tokenized)[0][0].cpu().float()

        vector = torch.clamp(prediction, min=0.0, max=1.0).tolist()
        obj['lisa_vector'] = vector
#         print('tagged in', time() - start_time, 'seconds')  # Print time taken
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

                if len(tagged_objects) % 50000 == 0:
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

    if num_workers > cpu_count():
        num_workers = cpu_count()

    print(f"Using {num_workers} workers for tagging")
    with Pool(num_workers, initializer=initialize_worker) as pool:
        pool.starmap(tag_partition, process_args)
    print("Finished multiprocessing pool")
