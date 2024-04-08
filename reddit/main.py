import json
import os
from tqdm import tqdm
import zstandard as zstd
import csv

from utils import get_file_handle, get_all_raw_files, word_count, dict_add

out_path = '/shared/3/projects/hiatus/tagged_data/long-reddit/'

index = 0

if __name__ == "__main__":
    all_files = get_all_raw_files()

    for file_path in all_files:
        author_subreddit_counts = {}
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]  # Remove file extension
        with get_file_handle(file_path) as file:
            out_file_path = os.path.join(out_path, file_name_without_ext + "_filtered.zst")

            with open(out_file_path, "wb") as outfile:
                cctx = zstd.ZstdCompressor()
                compressor = cctx.stream_writer(outfile)
                
                for line in tqdm(file):
                    doc = json.loads(line)
                    if word_count(doc["body"]) >= 250:
                        dict_add(author_subreddit_counts, doc["author"], doc["subreddit"])
                        
                        # Do whatever processing you need here
                        
                        # Example: Write the JSON object to the compressed file
                        compressor.write(json.dumps(doc).encode('utf-8'))
                        compressor.write(b'\n')  # Add a newline after each JSON object
                        index += 1

                # Flush and close the compressor
                compressor.flush(zstd.FLUSH_FRAME)
                compressor.close()

        # Potentially zip later?
        out_file_dir = os.path.join(out_path, file_name_without_ext)
        os.makedirs(out_file_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        # Open the file in append mode to avoid overwriting existing content
        with open(os.path.join(out_file_dir, "author_subreddit_count.tsv"), "a", newline="") as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            
            # Write data to the file
            for author, subreddit_count in author_subreddit_counts.items():
                for subreddit, count in subreddit_count.items():
                    writer.writerow([author, subreddit, count])
