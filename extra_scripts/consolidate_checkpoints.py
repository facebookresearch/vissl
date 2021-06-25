import argparse
import os

from extra_scripts.convert_sharded_checkpoint import (
    convert_checkpoint,
    setup_pathmanager,
)


def get_argument_parser():
    """
    List of arguments supported by the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        help="Starting iteration to consolidate",
    )
    parser.add_argument(
        "-f",
        "--finish",
        type=int,
        help="Ending iteration to consolidate",
    )
    parser.add_argument(
        "-e",
        "--every",
        type=int,
        help="Skip every N iterations.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input path to the sharded checkpoint.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path to the consolidated checkpoint.",
    )

    return parser


if __name__ == "__main__":
    """
    This is a thin wrapper around extra_scripts/convert_sharded checkpoint.py,
    designed to consolidate multiple checkpoints at once from a starting iteration,
    ending iteration, every N checkpoints.

    example usage:
    ```
    python extra_scripts/consolidate_checkpoints.py \
        -s 5000 -f 60000 -e 5000 \
        -i /path/to/sharded -o /path/to/consolidated
    ```
    """
    args = get_argument_parser().parse_args()

    mod_target = args.start % args.every
    for i in range(args.start, args.finish + 1):
        checkpoint_file_name = f"model_iteration{i}.torch"
        checkpoint_src = os.path.join(args.input, checkpoint_file_name)
        checkpoint_dest = os.path.join(args.output, checkpoint_file_name)

        if i % args.every == mod_target:
            setup_pathmanager()
            convert_checkpoint(checkpoint_src, checkpoint_dest, "consolidated")
            print(f"Consolidated { checkpoint_src } to { checkpoint_dest }.")
