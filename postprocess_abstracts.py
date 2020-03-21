from utils import configure_logging

import argparse
import logging
import os
import nltk
import random

from tqdm import tqdm
from itertools import zip_longest
from joblib import Parallel, delayed

N_LINES_PER_SLICE = int(1e6)


def main(options):
    configure_logging()

    os.makedirs(options.processed_dir, exist_ok=True)

    all_data = list()

    for file in tqdm(
        list(os.listdir(options.mesh_texts_dir)), desc="Reading input txt files ..."
    ):
        fpath = os.path.join(options.mesh_texts_dir, file)
        if os.path.isfile(fpath):
            with open(fpath, "r") as f:
                all_data.extend(f.read().splitlines())

    logging.info(f"Total number of sentences found : {len(all_data)}")

    logging.info(f"Removing duplicates")
    all_data = list(set(all_data))
    logging.info(f"Total number of sentences with duplicates removed : {len(all_data)}")

    logging.info("Shuffling all sentences")
    all_data = [(random.random(), line) for line in all_data]
    all_data.sort()
    all_data = [d[1] for d in all_data]

    all_sents_path = os.path.join(options.processed_dir, "all_sents.txt")
    with open(all_sents_path, "w") as f:
        for lines in tqdm(
            list(grouper(all_data, N_LINES_PER_SLICE)),
            desc=f"Writing raw data to {all_sents_path}",
        ):
            lines = filter(None, lines)
            f.write("\n".join(lines))

    tokenized_path = os.path.join(options.processed_dir, "tokenized.txt")
    with open(tokenized_path, "w") as f:
        for lines in tqdm(
            list(grouper(all_data, N_LINES_PER_SLICE)),
            desc=f"Writing tokenized data to {tokenized_path}",
        ):
            processed_slice = Parallel(n_jobs=-1, verbose=1)(
                process_line(line) for line in lines
            )
            processed_slice = filter(None, processed_slice)
            f.write("".join(processed_slice))


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


@delayed
def process_line(line):
    if line:
        return " ".join(nltk.word_tokenize(line)) + "\n"
    else:
        return None


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--mesh_texts_dir", type=str, default="./mesh_texts/")
    argument_parser.add_argument(
        "--processed_dir", type=str, default="./processed_texts/"
    )
    options = argument_parser.parse_args()
    main(options)
