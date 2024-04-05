# import gzip
# import os
# import csv
import gzip
import json
from utils import get_file_handle, word_count, dict_add

# Path to the directory containing .gz files
# directory_path = '/shared/2/datasets/reddit-dump-all/RC'

test_path = '/shared/2/datasets/reddit-dump-all/RC/RC_2013-05.bz2'
if __name__ == "__main__":
    # default dict of default dicts
    author_subreddit_counts = {}


    with get_file_handle(test_path) as file:
        for line in file:
            doc = json.loads(line.decode('utf-8'))
            if(True):
                dict_add(author_subreddit_counts, doc["author"], doc["subreddit"])
                print(author_subreddit_counts)
                break
            
            






