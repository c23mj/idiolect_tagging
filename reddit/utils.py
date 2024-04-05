import bz2
import io
import lzma
from glob import glob
from nltk.tokenize import word_tokenize
import zstandard as zstd

def word_count(text: str):
    return len(word_tokenize(text))

# helper function for adding into a dictionary. Defaultdicts aren't json serializable :(
def dict_add(dictionary: dict, outer_key, inner_key):
    if outer_key not in dictionary:
        dictionary[outer_key] = {}
    if inner_key not in dictionary[outer_key]:
        dictionary[outer_key][inner_key] = 0
    dictionary[outer_key][inner_key] += 1

"""
    Functions to work with the raw compressed Reddit data
"""
def get_file_handle(file_path):
    ext = file_path.split('.')[-1]

    if ext == "bz2":
        return bz2.open(file_path)
    elif ext == "xz":
        return lzma.open(file_path)
    elif ext == "zst":
        f = open(file_path, 'rb')
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        return text_stream

    raise AssertionError("Invalid extension for " + file_path + ". Expecting bz2 or xz file")


def get_all_raw_files():
    print("Reading in raw file names")
    files = glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    print("Total of {} files".format(len(files)))
    return files


