from utils import configure_logging

import argparse
import logging
import os
import pickle

from tqdm import tqdm
from Bio import Entrez

N_IDS_PER_REQUEST = 100000000


def collect_mesh(mesh_term, ids_dir):
    output_path = os.path.join(ids_dir, f"{mesh_term}.pck")

    if os.path.isfile(output_path):
        return

    handle = Entrez.esearch(db="pubmed", retmax=1, term=mesh_term, rettype="json")

    record = Entrez.read(handle)

    n_requests_needed = int(int(record["Count"]) / N_IDS_PER_REQUEST) + 1

    mesh_ids = []

    for i in tqdm(range(n_requests_needed)):
        handle = Entrez.esearch(
            db="pubmed",
            retmax=N_IDS_PER_REQUEST,
            retstart=i * N_IDS_PER_REQUEST,
            term=mesh_term,
            rettype="json",
        )

        mesh_ids.extend(Entrez.read(handle)["IdList"])

    with open(output_path, "wb") as f:
        pickle.dump(mesh_ids, f)


def main(options):
    configure_logging()

    Entrez.email = options.email

    logging.info(
        f"Beginning abstracts gathering for mesh terms located in file {options.mesh_terms_path}"
    )

    with open(options.mesh_terms_path, "r") as mesh_file:
        mesh_terms = mesh_file.read().splitlines()

    logging.info(f"The following mesh terms were retrieved {', '.join(mesh_terms)}")

    os.makedirs(options.ids_dir, exist_ok=True)

    for mesh_term in tqdm(mesh_terms, desc="Collecting ids by mesh terms..."):
        collect_mesh(mesh_term, options.ids_dir)

    logging.info("Done")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--mesh_terms_path", type=str, default="./mesh_terms.txt"
    )
    argument_parser.add_argument("--ids_dir", type=str, default="./mesh_ids/")
    argument_parser.add_argument(
        "--email", type=str, default="mathieu.godbout.3@ulaval.ca"
    )
    options = argument_parser.parse_args()
    main(options)
