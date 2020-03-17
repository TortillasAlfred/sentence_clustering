from utils import configure_logging

import argparse
import logging


def main(options):
    configure_logging()

    logging.info(
        f"Beginning abstracts gathering for mesh terms located in file {options.mesh_terms_path}"
    )

    with open(options.mesh_terms_path, "r") as mesh_file:
        mesh_terms = mesh_file.read().splitlines()

    logging.info(f"The following mesh terms were retrieved {', '.join(mesh_terms)}")

    logging.info("Done")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--mesh_terms_path", type=str, default="./mesh_terms.txt"
    )
    options = argument_parser.parse_args()
    main(options)
