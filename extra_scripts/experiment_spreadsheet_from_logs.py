# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import collections
import os
import pathlib
import re
import typing

import pandas as pd


def parse_log(log_path: str, args) -> dict:
    config = parse_config_from_log(log_path)
    # Check to make sure config not empty
    if config:
        config = flatten(config)
        if args.parse_date_time:
            date_time = None
            try:
                date_time = parse_date_time(
                    config[args.date_time_param],
                    args.date_time_pattern,
                    args.date_time_split_char,
                )
            except BaseException:
                pass
            if not date_time:
                print("Unable to parse date/time")
                date_time = [None, None]
            update_config_date_time(config, date_time)
    return config


def parse_config_from_log(log_path: str) -> dict:
    # String prepending beginning of config
    config_start_split_on = r"hydra_config.py: \d*: "
    # String at start of config
    config_start = "{'CHECKPOINT': "
    # String on final line of config
    config_end = "'VERBOSE': "
    config = ""
    # Flag to indicate the config portion of the log has been read
    config_finished = False

    # World size info from config is not reliable. Use the
    # String prepending beginning of world size info
    world_size_string = "WORLD_SIZE:"
    world_size_btwn = ("WORLD_SIZE:\t", "\n")
    world_size = None

    train_losses = []
    train_loss_str = "loss:"
    loss_string_btwn = ("loss: ", ";")

    latest_epoch = 0
    epoch_string = "[ep: "
    epoch_regex = r"(?<=\[ep: )\d{1,5}(?=\])"

    accuracies = {
        "train": {"string": "train_accuracy_list_meter", "values": []},
        "test": {"string": "test_accuracy_list_meter", "values": []},
    }

    with open(log_path) as reader:
        store_line = False
        # # There are some logs in which the config is printed multiple times.
        # # config_read_complete is used to avoid reading more than one config
        # # printing.
        # config_read_complete = False
        for line in reader:
            if not store_line:
                if world_size_string in line:
                    world_size = line
                if train_loss_str in line:
                    train_losses.append(line)
                for partition in accuracies.keys():
                    if accuracies[partition]["string"] in line:
                        accuracies[partition]["values"].append(line)
            if not config_finished:
                if config_start in line:
                    store_line = True
                    line = re.split(config_start_split_on, line)[1]
                if store_line:
                    config += line
                if config_end in line:
                    store_line = False
                    config_finished = True
            if epoch_string in line:
                epoch = re.search(epoch_regex, line)
                if epoch:
                    latest_epoch = int(epoch.group(0))

    if config:
        # Parse into dict
        try:
            config = ast.literal_eval(config)
            config = collections.OrderedDict(config)
        except BaseException:
            print("Unable to parse dictionary")
            config = {}
        # Add latest epoch to config
        config["latest_epoch"] = latest_epoch
        # Parse world size from string
        try:
            world_size = world_size.split(world_size_btwn[0])[1]
            world_size = world_size.split(world_size_btwn[1])[0]
            world_size = int(world_size)
            # Add to dict
            config["WORLD_SIZE"] = world_size
        except BaseException:
            print("Unable to parse world size")
        try:
            final_loss = train_losses[-1]
            final_loss = final_loss.split(loss_string_btwn[0])[1]
            final_loss = final_loss.split(loss_string_btwn[1])[0]
            config["final_train_loss"] = final_loss
        except BaseException:
            print("Unable to parse final training loss")
        for partition, partition_contents in accuracies.items():
            if partition_contents["values"]:
                try:
                    final_accuracy_string = partition_contents["values"][-1]
                    for top_string in ["top_1", "top_5"]:
                        acc = final_accuracy_string.split("value")[1].split(top_string)
                        acc = acc[1].split("0: ")[1]
                        acc = acc.split("}")[0]
                        param_str = f"final_{partition}_accuracy_{top_string}"
                        config[param_str] = float(acc)
                except BaseException:
                    print(f"Unable to parse final {partition} accuracy")
    else:
        print("No information parsed from log file")
        config = {}

    return config


def flatten(d: collections.abc.MutableMapping, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return collections.OrderedDict(items)


def parse_date_time(
    str_to_parse: str = None, pattern: str = None, split_char: str = None
):
    instances = re.findall(pattern, str_to_parse)
    if instances:
        date_time = instances[0].split(split_char)
        return date_time


def update_config_date_time(
    config: collections.OrderedDict, date_time: typing.Union[list, tuple]
):
    config["date"] = date_time[0]
    config["time"] = date_time[1]
    config.move_to_end("time", last=False)
    config.move_to_end("date", last=False)


def get_latest_checkpoint(directory: pathlib.PosixPath, args: argparse.Namespace):
    latest_checkpoint = None
    checkpoint_files = list(directory.glob(f"*{args.checkpoint_id_pattern}*"))
    if checkpoint_files:
        latest_checkpoint = 0
        for checkpoint_file in checkpoint_files:
            checkpoint_file = str(checkpoint_file).split("/")[-1]
            checkpoint_epoch = re.findall(
                args.checkpoint_extract_pattern, checkpoint_file
            )
            checkpoint_epoch = int(checkpoint_epoch[0])
            if checkpoint_epoch > latest_checkpoint:
                latest_checkpoint = checkpoint_epoch
        pass
    else:
        print("Unable to parse latest checkpoint information")
    return latest_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_directory",
        nargs="*",
        type=str,
        help="Directory or directories containing experiment "
        "run or subdirectories of runs",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default=os.getcwd(),
        help="Where to save output.",
    )
    parser.add_argument(
        "--output_name", type=str, default="experiments.txt", help="Output filename"
    )
    parser.add_argument(
        "--parse_date_time",
        type=bool,
        default=True,
        help="Parse date and time from config",
    )
    parser.add_argument(
        "--date_time_param",
        type=str,
        default="CHECKPOINT.DIR",
        help="config param from whose value the date and time will be parsed",
    )
    parser.add_argument(
        "--date_time_pattern",
        type=str,
        default="[0-9]{4}-[0-9][0-9]-[0-9][0-9]/[0-9][0-9]-[0-9][0-9]-[0-9][0-9]",
        help="Regex pattern for date and time format",
    )
    parser.add_argument(
        "--date_time_split_char",
        type=str,
        default="/",
        help="character to split date and time string into " "separate strings",
    )
    parser.add_argument(
        "--log_file_name_pattern",
        type=str,
        default="log.txt",
        help="pattern to match for log " "file names",
    )
    parser.add_argument(
        "--parse_checkpoint",
        type=bool,
        default=True,
        help="Parse # training epochs from checkpoint file",
    )
    parser.add_argument(
        "--checkpoint_id_pattern",
        type=str,
        default="_phase",
        help="pattern to match for " "checkpoint file names",
    )
    parser.add_argument(
        "--checkpoint_extract_pattern",
        type=str,
        default=r"phase([0-9]{1,4})\.torch",
        help="pattern to extract epoch # from checkpoint file name",
    )

    args = parser.parse_args()

    log_files = []
    for directory in args.root_directory:
        log_files.extend(
            list(pathlib.Path(directory).rglob(args.log_file_name_pattern))
        )

    configs_to_concat = []
    for f in log_files:
        did_not_add = True
        print(f"\nParsing {f}")
        config = parse_log(str(f), args)
        if args.parse_checkpoint:
            last_checkpoint = get_latest_checkpoint(f.parent, args)
            config["last_checkpoint_phase"] = last_checkpoint
        if args.parse_checkpoint and config["last_checkpoint_phase"]:
            configs_to_concat.append(config)
            did_not_add = False
        elif not args.parse_checkpoint:
            configs_to_concat.append(config)
            did_not_add = False
        if did_not_add:
            print(f"Did not add\n{f}\nto file")
        if not did_not_add:
            print(f"Added \n{f}\nto file")
    df = pd.DataFrame(configs_to_concat)
    # Sort columns
    df = df.reindex(sorted(df.columns), axis=1)
    # Move specific columns to beginning. Columns are listed here in reverse
    # order. The final item in the list will be the first column.
    prepend_columns = [
        "final_train_loss",
        "final_test_loss",
        "final_train_accuracy_top_1",
        "final_train_accuracy_top_5",
        "final_test_accuracy_top_1",
        "final_test_accuracy_top_5",
        "latest_epoch",
        "last_checkpoint_phase",
        "time",
        "date",
    ]
    for prepend_column in prepend_columns:
        if prepend_column in df.columns:
            df.insert(0, prepend_column, df.pop(prepend_column))
    output_full_path = os.path.join(args.output_directory, args.output_name)
    df.to_csv(output_full_path)
    print(f"Saved {output_full_path}")
