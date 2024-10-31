import argparse
import os

import zerohertzLib as zz


def main():
    logger.info(f"zerohertzLib version: {zz.__version__}")
    sphinx = os.path.dirname(__file__)
    if args.token is None:
        gh = zz.api.GitHub()
    else:
        gh = zz.api.GitHub(token=args.token)
    gh.release_note(sphinx_source_path=os.path.join(sphinx, "source"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token")
    args = parser.parse_args()
    logger = zz.logging.Logger("Release Note")
    main()
