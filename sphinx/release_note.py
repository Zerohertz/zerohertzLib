import argparse
import os

import zerohertzLib as zz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token")
    args = parser.parse_args()
    sphinx = os.path.dirname(__file__)
    if args.token is None:
        gh = zz.api.GitHub()
    else:
        gh = zz.api.GitHub(token=args.token)
    gh.release_note(sphinx_source_path=os.path.join(sphinx, "source"))
