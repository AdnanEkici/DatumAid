from __future__ import annotations

import importlib
import os
import shutil
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from copy import deepcopy
from glob import glob
from re import match
from typing import Callable

import cv2
import datumaro as dm
import pandas as pd
import polars as pl
from datumaro.components.annotation import AnnotationType
from datumaro.plugins.transforms import ProjectLabels
from datumaro.plugins.transforms import RandomSplit
from datumaro.plugins.transforms import Reindex
from datumaro.plugins.transforms import RemapLabels
from tqdm import tqdm

import datumaid
import datumaid.utils as utils
from datumaid import image_module  # noqa
from datumaid import selection_module  # noqa
from datumaid.api_handler import CVATHandler


class RandomSplitKeepDefaults(RandomSplit):
    """Inherited from original code for keeping the default subsets."""

    def __iter__(self):
        for i, item in enumerate(self._extractor):
            if item.subset in self._subsets:
                yield item
            else:
                yield self.wrap_item(item, subset=self._find_split(i))


class DatasetManager:
    temp_dir_default = os.sep + "tmp" + os.sep + "DatumAid" + os.sep

    def __init__(self, config_yml, **kwargs):
        self.config_yml = config_yml
        self.temp_dir = f"{self.temp_dir_default}{os.getpid()}" + os.sep
        if os.path.exists(self.temp_dir) and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.temp_dir_datasets = self.temp_dir + "datasets" + os.sep
        utils.get_logger().debug(f"default temp path is {self.temp_dir}")

    @classmethod
    def from_yaml(cls, config_yml, **kwargs):
        c = cls(config_yml, **kwargs)
        c.__import_datasets_from_yaml()
        utils.get_logger().debug("import from yaml success.")
        return c

    def __del__(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            utils.get_logger().debug(self.temp_dir + "directory deleted")

    def __err(self, error_str):
        utils.get_logger().critical(error_str)
        raise RuntimeError(error_str)

    def __prepare_dataset_files(self, path) -> str:
        tmp_path = self.temp_dir_datasets + path
        if os.path.exists(tmp_path):
            self.__err(f"The path {tmp_path} will be used for unziping the zipped datasets is not empty. " "Please check the path than retry!")
        if os.path.isdir(path):
            shutil.copytree(path, tmp_path)
        elif zipfile.is_zipfile(path):
            utils.extract_zipfile(tmp_path, path)
        elif tarfile.is_tarfile(path):
            with tarfile.TarFile(path, "r") as tar_ref:
                tar_ref.extractall(tmp_path)
        else:
            self.__err("unsupported dataset " + path)
        return tmp_path

    def __import_datasets_from_yaml(self):
        if self.config_yml is None:
            self.__err("config yaml is none")
        dataset_size = 0
        total_frame_size = 0
        for dataset in self.config_yml["datasets"]:
            dataset["datum_datasets"] = []
            dataset["tmp_path"] = []
            if "download" in dataset:
                api_handler = CVATHandler(dataset["download"]["url"], dataset["download"]["token"])
                utils.get_logger().info(
                    f"CVAT API has been started. Requests will be sent to {dataset['download']['url']} with token {dataset['download']['token']}"
                )
                export_type = api_handler.get_export_type(dataset["download"]["include_images"])
                export_path = dataset["download"]["download_path"]
                download_format = dataset["download"]["format"]

                if "project_id" in dataset["download"]:
                    for project_id in dataset["download"]["project_id"]:
                        dataset["path"] += api_handler.export_project(
                            project_id=project_id,
                            export_type=export_type,
                            export_path=export_path,
                            format=download_format,
                            exclude_list=dataset["download"]["exclude_tasks"],
                        )
                if "task_id" in dataset["download"]:
                    if "exclude_tasks" in dataset["download"]:
                        utils.get_logger().warning("'Exclude tasks' in config only works for project download.")
                    # has to be appended like this, or it will add each char of string.
                    dataset["path"] += [
                        api_handler.export_task(
                            task_id=task_id,
                            export_type=export_type,
                            export_path=export_path,
                            format=download_format,
                        )
                        for task_id in dataset["download"]["task_id"]
                    ]

            for path in dataset["path"]:
                tmp_path = self.__prepare_dataset_files(path)

                self.__dataset_type_checker(dataset, tmp_path)

                dataset["tmp_path"].append(tmp_path)

                datum_dataset = dm.Dataset.import_from(tmp_path, dataset["type"])

                utils.get_logger().warning(f"{tmp_path} dataset has {len(datum_dataset)} items.")

                for item in datum_dataset:
                    utils.fix_image_path(item)

                total_frame_size += len(datum_dataset)
                dataset["datum_datasets"].append(datum_dataset)
                dataset_size += 1
        utils.get_logger().warning(f"{dataset_size} datasets imported with total {total_frame_size} frames.")

    def __get_fun_from_exec_conf(self, exec_config):
        module_str = str(exec_config).split(".")[0]
        fun_str = str(exec_config).split(".")[1].split("(")[0]
        try:
            module = importlib.import_module("DatumAid.selection_module")
        except Exception:
            self.__err(f"module import error, module: {module_str}")
        try:
            selection_fn = getattr(module, fun_str)
        except Exception:
            self.__err(f"module function import error, module: {fun_str}")
        return selection_fn

    def __add_annotation_id_to_yaml(self, dataset, yaml):
        if "labels" in yaml:
            utils.add_annotation_ids(dataset, labels_yml=yaml["labels"], tag="label")
        if "bboxes" in yaml:
            utils.add_annotation_ids(dataset, labels_yml=yaml["bboxes"], tag="bbox")
        if "polygons" in yaml:
            utils.add_annotation_ids(dataset, labels_yml=yaml["polygons"], tag="polygon")

    def __get_video_path(self, cvat_annotation_path, mount_basedir="mount-data/cvat_data"):
        if not os.path.exists(mount_basedir):
            self.__err(f'Cvat data is not found in "{mount_basedir}"')

        tree = ET.parse(glob(cvat_annotation_path + "/*.xml")[0])
        meta_data = tree.getroot().find("meta")

        task_id = meta_data.find("task").find("id").text
        video_name = meta_data.find("task").find("source").text

        # Check given task_id
        if os.path.exists(os.path.join(mount_basedir, str(task_id), "raw", video_name)):
            return os.path.join(mount_basedir, str(task_id), "raw", video_name)

        # Apply shift and try again
        if os.path.exists(os.path.join(mount_basedir, str(int(task_id) + 5), "raw", video_name)):
            return os.path.join(mount_basedir, str(int(task_id) + 5), "raw", video_name)

        video_paths = glob(mount_basedir + f"/**/raw/{video_name}", recursive=True)

        if len(video_paths) >= 1:
            utils.get_logger().critical(
                f"{len(video_paths)} candidate is found for given task, one of them will be randomly returned. Check if the video is correct video"
            )

        return video_paths[-1]

    def __create_labels_dataframe(self, datum_dataset):
        """This method creates two lists of dictionaries that contain annotations for a dataset.
        If you call this method more than once, it will combine the dataset information with previous datasets.
        Dataset informations will be stored as a class variable.

        Parameters
        ----------
        datum_dataset : dataset

        """
        if not hasattr(self, "last_label_processed"):
            self.last_label_processed = 0  # Used for keep track of latest processed frame index
            self.last_ann_id = 0  # Used for creating incremental unique annotation id
            self.annotation_csv_dict_list = []
            self.attribute_csv_dict_list = []
            self.frame_to_annotation_idxs = {}  # Link between annotions and related image
            self.attribute_dict = {}  # Keeps all created attributes, main purpose is keeping attribute rows unique

        label_names_map = {}
        if len(list(datum_dataset._data._categories.keys())) > 0:
            label_names_map = datum_dataset._data._categories[list(datum_dataset._data._categories.keys())[0]]._indices
            label_names_map = {v: k for k, v in label_names_map.items()}

        processed_ids = [item.id for item in datum_dataset]
        real_idx = self.last_label_processed

        for idx in range(self.last_label_processed, len(self.config_yml["original_anns"])):
            org_item, gen_items = self.config_yml["original_anns"][idx], self.config_yml["polluted_anns"][idx]

            if org_item.id not in processed_ids:
                continue

            if len(gen_items) == 0:
                utils.get_logger().critical("No polluted item found!")
                continue

            for ann in org_item.annotations:
                attribute_idxs = []
                for att_name, att_value in ann.attributes.items():
                    if att_name == "label_id":
                        continue

                    if att_name == "value" and ann.label in label_names_map.keys():
                        att_name = label_names_map[ann.label]

                    attribute_key = f"{att_name}||{str(att_value)}"
                    if attribute_key not in self.attribute_dict:  # Unique value create new row
                        self.attribute_csv_dict_list.append({"id": len(self.attribute_dict), "name": att_name, "value": att_value})

                        attribute_idxs.append(len(self.attribute_dict))
                        self.attribute_dict[attribute_key] = len(self.attribute_dict)

                    else:
                        attribute_idxs.append(self.attribute_dict[attribute_key])

                row = {
                    "id": self.last_ann_id,
                    "attributes": attribute_idxs,
                    "type": ann.__class__.__name__,
                    "name": label_names_map[ann.label] if ann.label in label_names_map else None,
                    "points": ann.points if "Label" != ann.__class__.__name__ else [],
                }
                if real_idx in self.frame_to_annotation_idxs:
                    self.frame_to_annotation_idxs[real_idx].append(self.last_ann_id)
                else:
                    self.frame_to_annotation_idxs[real_idx] = [self.last_ann_id]

                self.last_ann_id += 1
                self.annotation_csv_dict_list.append(row)

            real_idx += 1

        self.last_label_processed = real_idx

    def __dataset_type_checker(self, dataset, tmp_path):
        if "type" not in dataset:
            dataset["type"] = dm.Dataset.detect(tmp_path)
        else:
            detected_type = dm.Dataset.detect(tmp_path)
            if detected_type != dataset["type"]:
                self.__err(f"given dataset type : '{dataset['type']}' not matched with detected : '{detected_type}'")

    def process_select(self, selects_yml, datum_dataset):
        for select_yml in selects_yml:
            self.__add_annotation_id_to_yaml(datum_dataset, select_yml)
            exec_str = select_yml["exec"]

            def selector(item, select_yml, exec_str):
                loc = {}
                loc["item"] = item
                loc["select_yml"] = select_yml
                loc["is_accepted"] = False
                exec_str__ = "is_accepted = " + exec_str
                exec(exec_str__, globals(), loc)
                returned_val = bool(loc["is_accepted"])
                return returned_val

            if len(datum_dataset) != 0:
                datum_dataset.select(lambda item: selector(item, select_yml, exec_str))
            len(datum_dataset)
            # TODO: check select not working immediately is a problem?

    def process_item_manipulation(self, item_manipulations_yml, datum_dataset):
        for item_yml in item_manipulations_yml:
            self.__add_annotation_id_to_yaml(datum_dataset, item_yml)

            exec_str = item_yml["exec"]
            for item in datum_dataset:
                loc = {}
                loc["item"] = item
                loc["item_yml"] = item_yml
                exec(exec_str, globals(), loc)

    def process_generator(self, dataset, datum_dataset):
        generator_yml = dataset["generator"]

        subsets = list(datum_dataset.subsets())
        if len(subsets) > 1:
            self.__err("In a dataset must be only one subset")
        default_format = subsets[0]

        if "original_anns" not in self.config_yml:
            self.config_yml["original_anns"] = []
            self.config_yml["polluted_anns"] = []

        generated_dataset = dm.Dataset(categories=datum_dataset.categories())

        pollute_back = generator_yml["pollute_back"]
        pollute_step = generator_yml["pollute_step"]

        if pollute_step <= 0 or pollute_back <= 0:
            self.__err("For generator pollute_back and pollute_step must greater than 1!")

        cvat_video_path = (
            ""
            if "tmp_path" not in dataset.keys()
            else self.__get_video_path(
                dataset["tmp_path"][dataset["datum_datasets"].index(datum_dataset)],
                dataset["generator"].get("cvat_data_path", "mount-data/cvat_data"),
            )
        )
        video_path = generator_yml["video_path"] if "video_path" in generator_yml else cvat_video_path

        # Create a directory for extracted frames
        frames_dir = self.temp_dir_datasets + "/" + os.path.dirname(video_path) + "/frames"
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if cap is None or not cap.isOpened():
            self.__err("Video is not found")

        ann_ids = [item.id for item in datum_dataset]
        ann_ids = list(filter(lambda id: match("frame_[0-9]+", id), ann_ids))

        n_zero_pad = len(ann_ids[0].split("_")[1])

        # Copy annotations and calculate needed frames
        polluted_frame_idxs = []
        for original_ann_id in sorted(ann_ids):
            frame_number = int(original_ann_id.split("_")[1])
            expected_frame_idxs = list(range(frame_number, max(frame_number - pollute_back - 1, -1), -pollute_step))
            original_ann = datum_dataset.get(original_ann_id, subset=default_format)

            polluted_anns_temp = []
            for expected_frame_idx in sorted(expected_frame_idxs):
                if expected_frame_idx in polluted_frame_idxs:
                    polluted_anns_temp.append(generated_dataset.get(f"frame_{str(expected_frame_idx).zfill(n_zero_pad)}", subset=default_format))
                    continue

                ann_id = f"frame_{str(expected_frame_idx).zfill(n_zero_pad)}"

                source_item_idx = frame_number
                for idx in polluted_frame_idxs:
                    if expected_frame_idx <= idx:
                        source_item_idx = idx

                source_item = datum_dataset.get(f"frame_{str(source_item_idx).zfill(n_zero_pad)}", subset=default_format)
                polluted_ann = original_ann if ann_id == original_ann.id else deepcopy(source_item)
                polluted_ann.id = ann_id

                generated_dataset.put(polluted_ann)
                if polluted_ann.id != original_ann.id:
                    polluted_anns_temp.append(polluted_ann)

            self.config_yml["original_anns"].append(original_ann)
            self.config_yml["polluted_anns"].append(polluted_anns_temp)

            polluted_frame_idxs += expected_frame_idxs

        # Remove duplicate idxs
        polluted_frame_idxs = sorted([item for idx, item in enumerate(polluted_frame_idxs) if item not in polluted_frame_idxs[:idx]])

        current_frame_idx = 0
        ret, frame = cap.read()
        while ret:
            if current_frame_idx in polluted_frame_idxs:
                image_path = frames_dir + f"/frame_{str(current_frame_idx).zfill(n_zero_pad)}.PNG"
                cv2.imwrite(image_path, frame)
                generated_dataset.get(f"frame_{str(current_frame_idx).zfill(n_zero_pad)}", subset=default_format).image = dm.Image(path=image_path)

                polluted_frame_idxs.remove(current_frame_idx)
            ret, frame = cap.read()
            current_frame_idx += 1

        if len(polluted_frame_idxs) > 0:
            self.__err("Video is shorter than given dataset!")

        datum_datasets = dataset["datum_datasets"]
        datum_datasets[datum_datasets.index(datum_dataset)] = generated_dataset
        return generated_dataset

    def process_split(self, dataset, datum_dataset):
        split_yml = dataset["split"]
        if "keep_defaults" in split_yml:
            splitter = RandomSplitKeepDefaults
        else:
            splitter = RandomSplit

        generated_dataset = dm.Dataset(categories=datum_dataset.categories())
        temp_dataset = deepcopy(datum_dataset)
        temp_dataset.transform(
            splitter,
            splits=split_yml["subset"].items(),
            seed=len(temp_dataset) / 2,
        )

        for temp_item, item in zip(temp_dataset, datum_dataset):
            item.subset = temp_item.subset
            generated_dataset.put(item)

        datum_datasets = dataset["datum_datasets"]
        datum_datasets[datum_datasets.index(datum_dataset)] = generated_dataset
        return generated_dataset

    def process_remap(self, remap_yml, datum_dataset):
        default = RemapLabels.DefaultAction.keep
        if "default" in remap_yml:
            default = eval("RemapLabels.DefaultAction." + remap_yml["default"])

        if "mapping" not in remap_yml:
            self.__err("mapping has to be under remap_labels:")

        tmp_dataset = RemapLabels(datum_dataset, mapping=remap_yml["mapping"], default=default)
        datum_dataset.categories()[AnnotationType.label] = tmp_dataset.categories()[AnnotationType.label]
        for tmp_item, item in zip(tmp_dataset, datum_dataset):
            item.annotations = tmp_item.annotations

    def process(self):
        last_data_start = 0
        for dataset in self.config_yml["datasets"]:
            for datum_dataset in dataset["datum_datasets"]:
                utils.get_logger().warning(f"dataset {datum_dataset.data_path} start proccess")
                if "selects" in dataset:
                    self.process_select(dataset["selects"], datum_dataset)
                    if len(datum_dataset) == 0:
                        continue
                    utils.get_logger().warning(f"dataset {datum_dataset.data_path} select completed")
                if "generator" in dataset:
                    datum_dataset = self.process_generator(dataset, datum_dataset)
                    utils.get_logger().warning(f"dataset {datum_dataset.data_path} generator completed")
                if "item_manipulations" in dataset:
                    self.process_item_manipulation(dataset["item_manipulations"], datum_dataset)
                    utils.get_logger().warning(f"dataset {datum_dataset.data_path} item_manipulations completed")
                if "split" in dataset:
                    datum_dataset = self.process_split(dataset, datum_dataset)
                    utils.get_logger().warning(f"dataset {datum_dataset.data_path} split completed")
                if "remap_labels" in dataset:
                    self.process_remap(dataset["remap_labels"], datum_dataset)
                    utils.get_logger().warning(f"dataset {datum_dataset.data_path} remap completed")

                if "generator" in dataset:
                    self.__create_labels_dataframe(datum_dataset)

                # TODO: get dataset stats here.
                if "project_labels" in self.config_yml:
                    # merge datasets
                    tmp_dataset = Reindex(datum_dataset, start=last_data_start)
                    for tmp_item, item in zip(tmp_dataset, datum_dataset):
                        item.id = tmp_item.id

                    utils.get_logger().debug(f"dataset {datum_dataset.data_path} reindex completed")

                    tmp_dataset = ProjectLabels(datum_dataset, dst_labels=self.config_yml["project_labels"])
                    datum_dataset.categories()[AnnotationType.label] = tmp_dataset.categories()[AnnotationType.label]
                    for tmp_item, item in zip(tmp_dataset, datum_dataset):
                        item.annotations = tmp_item.annotations

                    utils.get_logger().debug(f"dataset {datum_dataset.data_path} project_labels completed")

                    if last_data_start == 0:
                        self.config_yml["processed_dataset"] = datum_dataset
                    else:
                        for item in datum_dataset:
                            self.config_yml["processed_dataset"].put(item)
                    utils.get_logger().warning(f"dataset {datum_dataset.data_path} merge completed")
                    last_data_start += len(datum_dataset)

    def __export_custom_dataset(self):
        """Exports dataset to AI-Teams selected format which contains images and metadata csv files.

        Returns
        -------
        None
            Returns nothing
        """
        if "processed_dataset" not in self.config_yml:
            self.__err("processed dataset not found")

        # Create csv files
        base_csv_dict_list = []  # type: list
        frame_csv_dict_list = []  # type: list
        chunk_output = "chunk" in self.config_yml
        chunk_config = self.config_yml.get("chunk", {})
        chunked_output_directory_name = chunk_config.setdefault(
            "chunked_output_directory_name", "custom_dataset_export" + os.sep + "custom_dataset_chunked"
        )
        config_export_path = self.config_yml["export"]["path"]
        export_type = self.config_yml["export"]["type"]

        path = os.path.join(config_export_path, "custom_dataset")
        os.makedirs(path, exist_ok=True)

        seen_files = []

        def write(input_dict, target_dict, chunk=False):
            """Writes given images

            Parameters
            ----------
            input_dict : dict
                Input dictionary which contains absolute image paths as key and values asimages(ndarry). These images are original and previous frames
            target_dict : dict
                Dictionary which contains absolute image paths as key and values as images(ndarry). These images are target images.
            chunk : bool, optional
                flag that will written images also be chunked, by default False

            Returns
            -------
            str,str
                returns input images relative path, target images relative path.
            """

            input_keys = list(input_dict.keys())
            target_keys = list(target_dict.keys())

            [cv2.imwrite(f"{path+os.sep+key}", input_dict[key]) for key in input_keys]
            [cv2.imwrite(f"{path+os.sep+key}", target_dict[key]) for key in target_keys]

            if chunk:
                relocated_paths = DatasetManager.chunker(
                    target_directory=path,
                    target_extension_list=chunk_config.get("target_extension_list", ["_prev.PNG", ".PNG", "_target.PNG"]),
                    max_file_count=chunk_config.get("max_file_count", 500),
                    chunked_output_directory_name=chunked_output_directory_name,
                    seen_files=seen_files,
                )
                relocated_paths = [
                    os.path.relpath(relocated_path, (self.config_yml["export"]["path"] + os.sep + "custom_dataset_chunked"))
                    for relocated_path in relocated_paths
                ]

                return relocated_paths[: len(relocated_paths) - 1], [relocated_paths[len(relocated_paths) - 1]]
            return input_keys, target_keys

        real_idx = 0
        frame_idx = 0
        frame_ext = "PNG"

        max_number_of_digits = len(str(len(self.config_yml["original_anns"])))
        processed_ids = [item.id for item in self.config_yml["processed_dataset"]]
        for org_item, gen_item in zip(self.config_yml["original_anns"], self.config_yml["polluted_anns"]):
            org_id = org_item.id

            input_images_dict, target_images_dict = {}, {}  # type: dict
            if org_id not in processed_ids:
                continue

            if len(gen_item) == 0:
                utils.get_logger().critical("No polluted item found!")
                continue

            frame_name = f"frame_{str(real_idx).zfill(max_number_of_digits)}"

            mask_img = datumaid.utils.generate_mask(org_item)
            img_name = f"{frame_name}__target.{frame_ext}"
            target_images_dict[img_name] = mask_img

            img_name = frame_name + f".{frame_ext}"
            org_frame = cv2.imread(org_item.image.path)
            input_images_dict[img_name] = org_frame

            for idx, item in enumerate(gen_item):
                prev_frame = cv2.imread(item.image.path)
                img_name = f"{frame_name}__prev.{frame_ext}"
                input_images_dict[img_name] = prev_frame

            input_idxs = []
            target_idxs = []
            input_paths, target_paths = write(input_images_dict, target_images_dict, chunk=chunk_output)

            for idx, frame_path in enumerate(input_paths):
                frame = input_images_dict[list(input_images_dict.keys())[idx]]
                frame_row = {"id": frame_idx, "path": frame_path, "size": [frame.shape[1], frame.shape[0]]}
                frame_csv_dict_list.append(frame_row)
                input_idxs.append(frame_idx)
                frame_idx += 1

            for idx, frame_path in enumerate(target_paths):
                frame = target_images_dict[list(target_images_dict.keys())[idx]]
                frame_row = {"id": frame_idx, "path": frame_path, "size": [frame.shape[1], frame.shape[0]]}
                frame_csv_dict_list.append(frame_row)
                target_idxs.append(frame_idx)
                frame_idx += 1

            base_row = {"id": real_idx, "annotation_id": self.frame_to_annotation_idxs[real_idx], "inputs": input_idxs, "targets": target_idxs}
            base_csv_dict_list.append(base_row)

            real_idx += 1

        csv_save_location = chunked_output_directory_name if chunk_output else path
        if export_type == "parquet":
            pl.DataFrame(base_csv_dict_list).write_parquet(csv_save_location + os.sep + "base.parquet")
            pl.DataFrame(frame_csv_dict_list).write_parquet(csv_save_location + os.sep + "frames.parquet")
            pl.DataFrame(self.annotation_csv_dict_list).write_parquet(csv_save_location + os.sep + "annotation.parquet")
            pl.DataFrame(self.attribute_csv_dict_list).write_parquet(csv_save_location + os.sep + "attribute.parquet")
        elif export_type == "csv":
            pd.DataFrame(base_csv_dict_list).to_csv(csv_save_location + os.sep + "base.csv", index=False)
            pd.DataFrame(frame_csv_dict_list).to_csv(csv_save_location + os.sep + "frames.csv", index=False)
            pd.DataFrame(self.annotation_csv_dict_list).to_csv(csv_save_location + os.sep + "annotation.csv", index=False)
            pd.DataFrame(self.attribute_csv_dict_list).to_csv(csv_save_location + os.sep + "attribute.csv", index=False)
        else:
            raise RuntimeError(f"Export type is not supported: {export_type}")

        if len(os.listdir(path)) == 0:
            os.rmdir(path)
        return None

    def export(self):
        after_export = None
        if "export" not in self.config_yml:
            self.__err("export not defined in yaml")

        if "processed_dataset" not in self.config_yml:
            self.__err("processed dataset not found")

        utils.get_logger().debug("export started")
        export_yml = self.config_yml["export"]
        export_format = export_yml["format"]

        # TODO: check is there an not item!
        if export_yml["format"] == "custom_dataset":
            utils.get_logger().info("custom_dataset export type selected.")
            export_format = self.__export_custom_dataset()
        if export_format is not None:
            self.config_yml["processed_dataset"].export(save_dir=export_yml["path"], format=export_format, save_images=True)
            after_export() if after_export is not None else None
        if "check_duplicates" in self.config_yml:
            datumaid.DatasetManager.duplicate_finder(*utils.get_duplicate_finder_parameters(**self.config_yml["check_duplicates"]))
        if "chunk" in self.config_yml:
            datumaid.DatasetManager.chunker(**self.config_yml["chunk"])

        frame_count = len(self.config_yml["processed_dataset"])
        bbox_count = 0
        polgyon_count = 0
        label_count = 0
        for item in self.config_yml["processed_dataset"]:
            for annotation in item.annotations:
                if utils.is_bbox(annotation):
                    bbox_count += 1
                elif utils.is_label(annotation):
                    label_count += 1
                elif utils.is_polygon(annotation):
                    polgyon_count += 1

        utils.get_logger().info(
            f"export finished total frame count:{frame_count}\n"
            f"bbox count:{bbox_count}\n"
            f"polygon count:{polgyon_count}\n"
            f"label count:{label_count}\n"
        )

    @staticmethod
    def chunker(
        target_directory: str,
        target_extension_list: list,
        max_file_count=500,
        chunked_output_directory_name="chunked_output",
        safe_chunk=True,
        seen_files=None,
    ):
        """
        This code takes a target directory and a list of file extensions, and divides the files in the target
        directory into smaller 'chunks' based on the file extensions and a specified maximum file count per chunk.
        The chunked files can either be copied or moved to a specified output directory, depending on the value of
        the 'safe_chunk' flag.

        Parameters
        ----------
        target_directory : str
            specifies the directory that contains the files to be chunked.
        target_extension_list : list
            list of file extensions that the code should look for in the target directory.
        max_file_count : int , optional by default 500
            specifies the maximum number of files that should be included in each chunk.
        chunked_output_directory_name : str , optional, by default chunked_output
            specifies the name of the output directory where the chunked files will be placed.
        safe_chunk : boolean , optional, by default True
            flag determines whether the chunked files should be copied (True) or moved (False) from the target directory to the output directory.
        """

        utils.get_logger().debug(f"Files to be chunked: {target_directory}")
        utils.get_logger().debug(f"Target extensions: {target_extension_list}")
        utils.get_logger().debug(f"Max file count per chunk: {max_file_count}")
        utils.get_logger().debug(f"Output directory: {chunked_output_directory_name}")

        if seen_files is not None:
            utils.get_logger().debug("Chunking is off !")

            def move_file(source, destination):
                pass

        elif safe_chunk:
            seen_files = []
            utils.get_logger().debug("Safe chunking is on !")

            def move_file(source, destination):
                shutil.copy(source, destination)
                utils.get_logger().debug(f"Copied file {source} to {destination}")

        else:
            seen_files = []
            utils.get_logger().debug("Safe chunking is off !")

            def move_file(source, destination):
                shutil.move(source, destination)
                utils.get_logger().debug(f"Moved file {source} to {destination}")

        all_files_in_sub_directories = [
            item
            for item in sorted(glob(target_directory + "/**/*", recursive=True), key=utils.natural_keys)
            if any(item.endswith(target_extension) for target_extension in target_extension_list) and item not in seen_files
        ]
        seen_files.extend(all_files_in_sub_directories)

        max_target_file_count_in_chunk = max_file_count // len(target_extension_list)
        file_count = 0
        chunk_name_begin = 0
        chunk_name_end = chunk_name_begin + max_target_file_count_in_chunk - 1
        chunk_dir_name = str(chunk_name_begin) + "_" + str(chunk_name_end)
        relocated_path_list = []
        current_output_dir = os.path.dirname(all_files_in_sub_directories[0]).replace(target_directory, chunked_output_directory_name)

        def create_new_chunk_folder(chunk_name_begin, chunk_name_end, chunk_dir_name):
            chunk_name_begin += max_target_file_count_in_chunk
            chunk_name_end += max_target_file_count_in_chunk
            chunk_dir_name = str(chunk_name_begin) + "_" + str(chunk_name_end)
            os.makedirs(os.path.join(current_output_dir, chunk_dir_name), exist_ok=True)
            return 0, chunk_name_begin, chunk_name_end, chunk_dir_name

        #  If there is a chunk directory. If chunk directory is full create new chunk directory else continue to fill that chunk directory.
        if os.path.exists(current_output_dir):
            chunk_folders = [path for path in sorted(glob(os.path.join(current_output_dir, "*"), recursive=True)) if os.path.isdir(path)]
            for chunk_folder in chunk_folders:
                file_count = len(os.listdir(chunk_folder))
                if file_count == max_target_file_count_in_chunk * len(target_extension_list):
                    file_count, chunk_name_begin, chunk_name_end, chunk_dir_name = create_new_chunk_folder(
                        chunk_name_begin, chunk_name_end, chunk_dir_name
                    )

        for file in all_files_in_sub_directories:
            if current_output_dir != os.path.dirname(file).replace(target_directory, chunked_output_directory_name):
                chunk_name_begin = 0
                chunk_name_end = chunk_name_begin + max_target_file_count_in_chunk - 1
                chunk_dir_name = str(chunk_name_begin) + "_" + str(chunk_name_end)
                file_count = 0
                current_output_dir = os.path.dirname(file).replace(target_directory, chunked_output_directory_name)
            os.makedirs(os.path.join(current_output_dir, chunk_dir_name), exist_ok=True)

            move_file(file, os.path.join(current_output_dir, chunk_dir_name))
            relocated_path_list.append(os.path.join(current_output_dir, chunk_dir_name, os.path.basename(file)))

            file_count = file_count + 1
            if file_count == max_target_file_count_in_chunk * len(target_extension_list):
                file_count, chunk_name_begin, chunk_name_end, chunk_dir_name = create_new_chunk_folder(
                    chunk_name_begin, chunk_name_end, chunk_dir_name
                )

        utils.get_logger().debug("Chunking is done !")
        return relocated_path_list

    @staticmethod
    def duplicate_finder(target_directory: str, hash_function: Callable, duplicate_image_folder="duplicate_images"):
        """Takes target directory path and hash function and finds all duplicate
         images by defined hash function if a duplicate image found than that duplicate
         image will be moved to specified location.

        Parameters
        ----------
        target_directory : str
             specifies the directory that contains the files in which to find duplicates.
        hash_function : function
             specifies the hash function for the images.
        """

        utils.get_logger().info(f"Target directory: {target_directory}")
        utils.get_logger().info(f"Hash function: {hash_function.__name__}")
        utils.get_logger().info(f"Duplicate image folder: {duplicate_image_folder}")

        all_images = utils.get_images(target_directory=target_directory)
        all_images_and_hashes = [(image, hash_function(image)) for image in tqdm(all_images, desc="Hashing images")]
        all_images_and_hashes.sort(key=lambda x: x[1])

        for i in tqdm(range(len(all_images_and_hashes) - 1), desc="Duplicates are finding"):
            if all_images_and_hashes[i][1] == all_images_and_hashes[i + 1][1]:
                os.makedirs(duplicate_image_folder, exist_ok=True)
                utils.get_logger().warning(
                    f"{all_images_and_hashes[i][0]} and {all_images_and_hashes[i+1][0]} are same moving "
                    f"{all_images_and_hashes[i][0]} to {duplicate_image_folder} folder"
                )
                shutil.move(all_images_and_hashes[i][0], duplicate_image_folder + os.sep + os.path.basename(all_images_and_hashes[i][0]))
