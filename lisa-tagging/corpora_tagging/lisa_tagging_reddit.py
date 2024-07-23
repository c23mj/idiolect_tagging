import os
import jsonlines

from glob import glob
from multiprocessing import Pool
import torch

# FIGURE OUT THESE IMPORTS
from tokenization_enc_t5 import EncT5Tokenizer
from modeling_enc_t5 import EncT5ForSequenceClassification

tokenizer = EncT5Tokenizer.from_pretrained("t5-base")
model = EncT5ForSequenceClassification.from_pretrained("/home/oyadav/idolect/stylegenome_lisa_sfam/lisa_checkpoint", num_labels=768, problem_type="regression")

def tag_partitions(input_directory, output_directory, num_workers, default_niceness=20):
    process_args = build_process_args(input_directory, output_directory)

    def set_niceness():
        os.nice(default_niceness)

    with Pool(num_workers, initializer=set_niceness) as pool:
        pool.starmap(tag_partition, process_args)


def tag_lisa(obj):
    # This will return a 768-dimensional LISA vector for the text, where each element is a score from 0-1, where the score corresponds to each concept in "lisa_dimensions.json".
    tokenized = tokenizer(
        [obj['body']],
        truncation=True, max_length=512, padding=True, return_tensors="pt"
    )
    prediction = model.forward(**tokenized)[0][0].cpu().detach().float()
    vector = torch.clamp(prediction, min=0.0, max=1.0).numpy().tolist()
    del obj['encodings']
    obj['lisa_vector'] = vector
    return obj


def tag_partition(input_file, output_file):
    print(f"Tagging file {input_file}\n")
    tagged_objects = []

    with jsonlines.open(input_file) as reader:
        for obj in reader:
            tagged_object = tag_lisa(obj)
            tagged_objects.append(tagged_object)

            if len(tagged_objects) % 5000 == 0:
                append_chunk(output_file, tagged_objects)
                tagged_objects = []

    if tagged_objects:
        append_chunk(output_file, tagged_objects)


def build_process_args(input_directory, output_directory):
    partition_files = glob(f"{input_directory}/*.jsonl")
    print(input_directory)
    process_args = []

    for fp in partition_files:
        fname = fp.rsplit('/', 1)[-1].replace('.jsonl', '') + '-tagged.jsonl'
        out = f"{output_directory}{fname}"
        process_args.append((fp, out))

    return process_args

def append_chunk(output_file, tagged_objects):
    with jsonlines.open(output_file, mode='a') as writer:
        writer.write_all(tagged_objects)
