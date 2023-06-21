from __future__ import annotations

import sys

import fire
import yaml

import datumaid
import datumaid.utils as utils
from datumaid.version import VERSION


def read_yaml(path):
    """read the yaml from path

    Parameters
    ----------
    path : string
        yml file path

    Returns
    -------
    dictionary
        yaml config dict
    """
    with open(path) as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)


def show_version(**kwargs):
    print(VERSION)


def process_dataset(type, config_path=None, config=None, **kwargs):
    # config or yaml
    if config_path:
        curr_config = read_yaml(config_path)
        if config:
            curr_config.update(config)
    elif config:
        curr_config = yaml.safe_load(config)
    else:
        utils.get_logger().error("config not found. check config_path or config parameters!")
        exit(0)

    if type == "chunk":
        utils.get_logger().info("Chunking has been started.")
        datumaid.DatasetManager.chunker(**curr_config["chunk"])
        return
    elif type == "find_duplicate":
        datumaid.DatasetManager.duplicate_finder(**curr_config["check_duplicates"])
        return

    dmanager = datumaid.DatasetManager.from_yaml(curr_config)

    # process
    if type == "show":
        print(dmanager.config_yml)
    elif type == "process" or type == "inspect":
        dmanager.process()
        # TODO: add datumaro stats
        if type == "process":
            dmanager.export()
    else:
        utils.get_logger().fatal("unsupported dataset type command. use DatumAid dataset <show/inspect/process/chunk>")


app = {
    "version": show_version,
    "dataset": process_dataset,
}


def main(args=None):
    fire.Fire(app)


if __name__ == "__main__":
    sys.exit(main())
