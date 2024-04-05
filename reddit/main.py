import json
from utils import get_file_handle, get_all_raw_files, word_count, dict_add

out_path = '/shared/3/projects/hiatus/tagged_data/long-reddit/'

if __name__ == "__main__":
    all_files = get_all_raw_files()
    author_subreddit_counts = {}

    for file_path in all_files:
        with get_file_handle(file_path) as file:
            for line in file:
                doc = json.loads(line.decode('utf-8'))
                if(word_count(doc["body"]) >= 250):
                    dict_add(author_subreddit_counts, doc["author"], doc["subreddit"])
                    pass # some sort of writing, some sort of saving, some sort of checkpointing.


