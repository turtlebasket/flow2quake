#!/usr/bin/env python

"""
dm: Data Manager
"""

from argparse import ArgumentParser
import datetime
import json
import random
import re
import subprocess
import os
import logging
import hashlib

DATA_DIRNAME = "data"
OUTPUT_DIRNAME = "data-out"
REPO_DIR, FILENAME = os.path.split(os.path.abspath(__file__))
CONFIG_FILENAME = f"{REPO_DIR}/.dmconfig.json"
LOG_FILENAME = f"{REPO_DIR}/.dm.log"
REMOTE_HOST = "tecto.gps.caltech.edu"
REMOTE_DIR = "/export/data1/flow2quake"
RSYNC_FLAGS = [
    "-a",
    "--info=progress2",
    "--exclude",
    ".gitkeep",
    "--exclude",
    "__pycache__",
    "--exclude",
    ".DS_Store",
]

logger = logging.getLogger("dm")
logger.addHandler(logging.FileHandler(LOG_FILENAME))
parser = ArgumentParser(
    prog="dm",
    description=f"Data Manager utility for Flow2Quake ({REPO_DIR})",
    add_help=True,
)
subparsers = parser.add_subparsers(dest="action", title="subcommands")
subparsers.required = True
known_cases = os.listdir("./flow2quake/cases")

if os.path.isfile(CONFIG_FILENAME):
    with open(CONFIG_FILENAME, "r") as file:
        config = json.load(file)
else:
    config = {}

upload_data_parser = subparsers.add_parser(
    "upload-data", help="Upload input data for an individual case"
)
download_data_parser = subparsers.add_parser(
    "download-data", help="Download input data for an individual case"
)
upload_output_parser = subparsers.add_parser(
    "upload-output",
    help="Upload most recent outputs (plots, etc) for an individual case",
)
download_output_parser = subparsers.add_parser(
    "download-output",
    help="Download someone's outputs (plots, etc) for an individual case",
)
identify_parser = subparsers.add_parser(
    "identify", help="Set your name for upload entries"
)
status_parser = subparsers.add_parser("status", help="Check remote store status")


for sp in [
    upload_output_parser,
    download_output_parser,
    upload_data_parser,
    download_data_parser,
]:
    sp.add_argument(
        "-y",
        "--yes-to-all",
        action="store_true",
        help="Automatically confirm all y/n dialogs",
    )
    sp.add_argument(
        dest="case",
        metavar="CASE",
        help="Case being operated on (e.g. groningen, arbitrary)",
        action="store",
        choices=known_cases,
    )

download_output_parser.add_argument(
    "-i", "--id", action="store", help="ID of run output upload to restore"
)

identify_parser.add_argument(
    "-n", "--name", help="Name you will use", action="store", type=str
)

args = parser.parse_args()


def set_name(name=None):
    global config
    if not name:
        name = input('Enter your name - preferably "First Last": ')
    if not re.match(r"^[a-zA-Z ]+$", name):
        print("Name must consist of letters and spaces.")
        exit(1)
    config["name"] = name.replace("_", " ")


def confirm(msg=None):
    q = f'{"NOTICE: " + msg + " " if msg else ""}Do you want to proceed? [y/n] '
    if not (getattr(args, "yes_to_all", False) or input(q).lower() == "y"):
        print("Cancelled by user")
        exit(0)


def create_case_dir(case_name):
    remote_cmd = ";".join(
        [
            f"mkdir -p {REMOTE_DIR}/{case_name}",
            f"mkdir -p {REMOTE_DIR}/{case_name}/data",
            f"mkdir -p {REMOTE_DIR}/{case_name}/outputs",
        ]
    )
    completed = subprocess.run(
        ["ssh", "-t", REMOTE_HOST, remote_cmd],
        capture_output=True,
    )
    if completed.returncode != 0:
        print(f"Failed to create remote directory for {case_name}:")
        print(completed.stdout.decode("utf-8"))
        exit(1)


def create_new_output_dir(case_name):
    """
    :param case_name: Name of case to create upload directory for
    :returns: Tuple containing (upload id, output directory)
    """
    utcnow = datetime.datetime.now(datetime.timezone.utc)

    time_created = utcnow.isoformat().split(".")[0]
    author = config["name"]
    output_id = hashlib.shake_256(
        (str(utcnow.timestamp()) + str(random.randbytes(20))).encode()
    ).hexdigest(5)

    output_dir = (
        f"{REMOTE_DIR}/{case_name}/outputs/{time_created}_{author}_{output_id}".replace(
            " ", "-"
        )
    )
    completed = subprocess.run(
        ["ssh", "-t", REMOTE_HOST, f"mkdir -p {output_dir}"],
        capture_output=True,
    )
    if completed.returncode != 0:
        print(f"Failed to create new remote output entry for {case_name}:")
        print("command: ", " ".join(completed.args))
        print(completed.stdout.decode("utf-8"))
        exit(1)
    return (output_id, output_dir)


def remove_empty_remote_dir(dir):
    completed = subprocess.run(
        ["ssh", "-t", REMOTE_HOST, f"rmdir {dir}"],
        capture_output=True,
    )
    if completed.returncode != 0:
        print("Failed to remove empty remote directory:")
        print("command: ", " ".join(completed.args))
        print(completed.stdout.decode("utf-8"))
        exit(1)


try:
    match args.action:
        case "status":
            print("hi")

        case "upload-output":
            confirm("This will create a new run-output entry in the remote archive.")
            if "name" not in config.keys():
                set_name()
                print("Identity set. You can run `dm identify` to overwrite it.")
            print(f"Creating new upload...")
            src = f"flow2quake/cases/{args.case}/{OUTPUT_DIRNAME}/"
            output_id, dest_dir = create_new_output_dir(args.case)
            dest = f"{REMOTE_HOST}:{dest_dir}"
            completed = subprocess.run(["rsync", *RSYNC_FLAGS, "--", src, dest])
            if completed.returncode != 0:
                remove_empty_remote_dir(dest)
                print(f"Upload failed (command: {' '.join(completed.args)})")
                exit(1)
            print("Done")

        case "download-output":
            print("Not implemented")
            exit(1)

            output_id = args.id or input("Enter ID of output to restore: ")
            src = f"{REMOTE_HOST}:{REMOTE_DIR}/{args.case}"
            dest = f"flow2quake/cases/{args.case}/{OUTPUT_DIRNAME}"

        case "upload-data":
            src = f"flow2quake/cases/{args.case}/{DATA_DIRNAME}/"  # trailing slash: copy contents, not directory itself
            dest = f"{REMOTE_HOST}:{REMOTE_DIR}/{args.case}/{DATA_DIRNAME}"
            create_case_dir(args.case)
            confirm(
                f"This will upload your `{args.case}.{DATA_DIRNAME}` folder and will overwrite files on the remote archive."
            )
            print("Starting rsync...")
            completed = subprocess.run(["rsync", *RSYNC_FLAGS, "--", src, dest])
            if completed.returncode != 0:
                print("command: ", " ".join(completed.args))
                print(completed.stderr.decode("utf-8"))
                exit(1)
            print("Done")

        case "download-data":
            src = f"{REMOTE_HOST}:{REMOTE_DIR}/{args.case}/{DATA_DIRNAME}/"
            dest = f"flow2quake/cases/{args.case}/{DATA_DIRNAME}"
            create_case_dir(args.case)
            confirm(
                f"This will download from the remote archive and overwrite files in your `{args.case}.{DATA_DIRNAME}` folder."
            )
            print("Starting rsync...")
            completed = subprocess.run(["rsync", *RSYNC_FLAGS, "--", src, dest])
            if completed.returncode != 0:
                print("command: ", " ".join(completed.args))
                print(completed.stderr.decode("utf-8"))
                exit(1)
            print("Done")

        case "identify":
            set_name(args.name)
            print("Identity set. You can run `dm identify` again to overwrite it.")

    with open(CONFIG_FILENAME, "w") as file:
        json.dump(config, file)

except (KeyboardInterrupt, EOFError) as e:
    print("\nAborted")
