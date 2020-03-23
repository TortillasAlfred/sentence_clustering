from utils import configure_logging

import fasttext
import argparse
import logging
import os

from tqdm import tqdm

N_VALID_LINES = int(1e6)


def main(options):
    configure_logging()

    # Train the thing
    model = fasttext.train_unsupervised(options.source_file, thread=options.n_threads)

    # Save
    model.save(options.save_path)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--source_file", type=str, default="./processed/tokenized.txt"
    )
    argument_parser.add_argument("--save_path", type=str, default="fasttext_model.bin")
    argument_parser.add_argument("--n_threads", type=int, default=8)
    options = argument_parser.parse_args()
    main(options)
