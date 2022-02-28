"""Console script for depthai_lightning."""
import argparse
import sys


def main():
    """Console script for depthai_lightning."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "depthai_lightning.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
