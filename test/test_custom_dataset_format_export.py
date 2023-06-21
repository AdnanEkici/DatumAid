from __future__ import annotations

import os
import shutil
import unittest

import cv2
import datumaro as dm
import pandas as pd
import yaml
from datumaro.components.annotation import AnnotationType

from datumaid import DatasetManager
from test import utils

# TODO:
# [] These tests need to seperated. csv_test - chunk_test etc.
# [] Tests need more variations.


class CustomDatasetFormatExport(unittest.TestCase):
    test_dir = os.sep + "tmp" + os.sep + "test_custom_dataset_export"

    def setUp(self):
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        self.video_name = utils.create_sample_video(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_custom_dataset_format_export(self):
        test_config = yaml.safe_load(
            f"""
            datasets:
              - dataset:
                selects:
                  - select:
                    polygons:
                      - polygon: stickman
                        attr:
                          animation_quality: good
                    bboxes:
                      - bbox: stickman
                        attr:
                          animation_quality: good

                    exec: selection_module.any_of_annotations(item, select_yml)

                generator:
                  pollute_back: 4
                  pollute_step: 4
                  video_path: {self.video_name}

                remap_labels:
                  default: keep
                  mapping:
                    stickman: stickman
                    motion_captured:

                split:
                  subset:
                    train: 1

            project_labels: [stickman]

            export:
              format: custom_dataset
              type: csv
              path: {self.test_dir}

            chunk:
              target_directory: /tmp/test_custom_dataset_export/custom_dataset
              target_extension_list: ["_prev.PNG", ".PNG", "_target.PNG"]
              chunked_output_directory_name: /tmp/test_custom_dataset_export/custom_dataset_chunked
              safe_chunk: True
            """
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[
                        dm.Polygon(
                            [0, 0, 0, 50, 50, 0, 0, 0],
                            id=2,
                            label=1,
                            attributes={"animation_quality": "good"},
                        ),
                        dm.Label(id=3, label=2, attributes={"occluded": False}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Polygon(
                            [0, 0, 0, 50, 50, 0, 0, 0],
                            id=2,
                            label=1,
                            attributes={"animation_quality": "good"},
                        ),
                        dm.Label(id=3, label=0, attributes={"occluded": True, "animation_quality": "nice"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            50,
                            50,
                            id=2,
                            label=1,
                            attributes={"animation_quality": "good", "visibility": "bad"},
                        ),
                        dm.Label(id=3, label=3, attributes={"is_running": "harmful"}),
                        dm.Label(id=3, label=2, attributes={"animation_quality": "nice"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000020",
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            100,
                            100,
                            id=2,
                            label=1,
                            attributes={"animation_quality": "good"},
                        ),
                        dm.Label(id=3, label=2, attributes={"flip_book_animation": True, "custom": "thing"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000024",
                    annotations=[dm.Label(id=1, label=0)],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["motion_captured", "stickman", "scene_forest", "mask"],
                    ["custom", "flip_book_animation", "occluded", "animation_quality", "is_running"],
                )
            },
        )

        datum_dataset_org2 = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Polygon(
                            [0, 0, 0, 80, 80, 70, 60, 0],
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                        dm.Label(id=3, label=2, attributes={"occluded": False}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000016",
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            0,
                            50,
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                        dm.Bbox(
                            0,
                            0,
                            180,
                            50,
                            id=3,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                        dm.Label(id=4, label=3, attributes={"occluded": True}),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman", "scene_forest", "mask", "motion_captured"],
                    ["custom", "flip_book_animation", "occluded", "animation_quality", "is_running"],
                )
            },
        )

        chunk_output = "chunk" in test_config
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org, datum_dataset_org2]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1, f"Expected only 1 dataset but got {len(d_manager.config_yml['datasets'])}")
        self.assertEqual(
            len(d_manager.config_yml["datasets"][0]["datum_datasets"]),
            2,
            f"Expected only 2 datumdataset but got {len(d_manager.config_yml['datasets'][0]['datum_datasets'])}",
        )
        self.assertEqual(
            len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]),
            5,
            f" Expected only 5 items but got {len(d_manager.config_yml['datasets'][0]['datum_datasets'][0])}",
        )
        self.assertEqual(
            len(d_manager.config_yml["datasets"][0]["datum_datasets"][1]),
            2,
            f"Expected only 2 items but got{len(d_manager.config_yml['datasets'][0]['datum_datasets'][1])}",
        )

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 10)
        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="0",
                    annotations=[
                        dm.Polygon(
                            [0, 0, 0, 50, 50, 0, 0, 0],
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                    ],
                    subset="train",
                ),
                dm.DatasetItem(
                    id="1",
                    annotations=[
                        dm.Polygon(
                            [0, 0, 0, 50, 50, 0, 0, 0],
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                    ],
                    subset="train",
                ),
                dm.DatasetItem(
                    id="2",
                    annotations=[
                        dm.Polygon(
                            [0, 0, 0, 50, 50, 0, 0, 0],
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                    ],
                    subset="train",
                ),
                dm.DatasetItem(
                    id="3",
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            50,
                            50,
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good", "visibility": "bad"},
                        ),
                    ],
                    subset="train",
                ),
                dm.DatasetItem(
                    id="4",
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            100,
                            100,
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                    ],
                    subset="train",
                ),
                dm.DatasetItem(
                    id="5",
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            100,
                            100,
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                    ],
                    subset="train",
                ),
                dm.DatasetItem(
                    id="6",
                    annotations=[
                        dm.Polygon(
                            [0, 0, 0, 80, 80, 70, 60, 0],
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                    ],
                    subset="train",
                ),
                dm.DatasetItem(
                    id="7",
                    annotations=[
                        dm.Polygon(
                            [0, 0, 0, 80, 80, 70, 60, 0],
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                    ],
                    subset="train",
                ),
                dm.DatasetItem(
                    id="8",
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            0,
                            50,
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                        dm.Bbox(
                            0,
                            0,
                            180,
                            50,
                            id=3,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                    ],
                    subset="train",
                ),
                dm.DatasetItem(
                    id="9",
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            0,
                            50,
                            id=2,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                        dm.Bbox(
                            0,
                            0,
                            180,
                            50,
                            id=3,
                            label=0,
                            attributes={"animation_quality": "good"},
                        ),
                    ],
                    subset="train",
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman"],
                    ["custom", "flip_book_animation", "occluded", "animation_quality", "is_running"],
                )
            },
        )

        utils.compare_datumaro_datasets(self, d_manager.config_yml["processed_dataset"], datum_dataset_result)

        d_manager.export()

        export_path = frame_path_check = self.test_dir + os.sep + "custom_dataset"
        if chunk_output:
            frame_path_check = export_path + "_chunked" + os.sep + "0_165"
            export_path = export_path + "_chunked"

        self.assertTrue(os.path.isdir(export_path), f"There is no directory called {export_path}")

        self.assertTrue(os.path.exists(frame_path_check + "/frame_0.PNG"), f"There is no file called {frame_path_check}/frame_0.PNG")
        self.assertTrue(os.path.exists(frame_path_check + "/frame_0__prev.PNG"), f"There is no file called {frame_path_check}/frame_0__prev.PNG")
        self.assertTrue(os.path.exists(frame_path_check + "/frame_0__target.PNG"), f"There is no file called {frame_path_check}/frame_0__target.PNG")

        self.assertTrue(utils.compare_sample_video_frame_ids(cv2.imread(frame_path_check + "/frame_0.PNG"), 8), "Wrong frame extracted!")
        self.assertTrue(utils.compare_sample_video_frame_ids(cv2.imread(frame_path_check + "/frame_0__prev.PNG"), 4), "Wrong frame extracted!")

        self.assertTrue(os.path.exists(frame_path_check + "/frame_1.PNG"), f"There is no file called {frame_path_check}/frame_1.PNG")
        self.assertTrue(os.path.exists(frame_path_check + "/frame_1__prev.PNG"), f"There is no file called {frame_path_check}/frame_1__prev.PNG")
        self.assertTrue(os.path.exists(frame_path_check + "/frame_1__target.PNG"), f"There is no file called {frame_path_check}/frame_1__target.PNG")

        self.assertTrue(utils.compare_sample_video_frame_ids(cv2.imread(frame_path_check + "/frame_1.PNG"), 12), "Wrong frame extracted!")
        self.assertTrue(utils.compare_sample_video_frame_ids(cv2.imread(frame_path_check + "/frame_1__prev.PNG"), 8), "Wrong frame extracted!")

        self.assertTrue(os.path.exists(frame_path_check + "/frame_2.PNG"), f"There is no file called {frame_path_check}/frame_2.PNG")
        self.assertTrue(os.path.exists(frame_path_check + "/frame_2__prev.PNG"), f"There is no file called {frame_path_check}/frame_2__prev.PNG")
        self.assertTrue(os.path.exists(frame_path_check + "/frame_2__target.PNG"), f"There is no file called {frame_path_check}/frame_2__target.PNG")

        self.assertTrue(utils.compare_sample_video_frame_ids(cv2.imread(frame_path_check + "/frame_2.PNG"), 20), "Wrong frame extracted!")
        self.assertTrue(utils.compare_sample_video_frame_ids(cv2.imread(frame_path_check + "/frame_2__prev.PNG"), 16), "Wrong frame extracted!")

        base_csv_file_path = os.path.join(export_path, "base.csv")
        self.assertTrue(os.path.exists(base_csv_file_path), f"Csv file is not found {base_csv_file_path}")
        base_csv = pd.read_csv(base_csv_file_path)

        relative_path = "0_165" + os.sep if chunk_output else ""

        expected_base_csv = pd.DataFrame(
            [
                {
                    "id": 0,
                    "annotation_id": "[0]",
                    "inputs": "[0, 1]",
                    "targets": "[2]",
                },
                {
                    "id": 1,
                    "annotation_id": "[1, 2, 3]",
                    "inputs": "[3, 4]",
                    "targets": "[5]",
                },
                {
                    "id": 2,
                    "annotation_id": "[4, 5]",
                    "inputs": "[6, 7]",
                    "targets": "[8]",
                },
                {
                    "id": 3,
                    "annotation_id": "[6, 7]",
                    "inputs": "[9, 10]",
                    "targets": "[11]",
                },
                {
                    "id": 4,
                    "annotation_id": "[8, 9]",
                    "inputs": "[12, 13]",
                    "targets": "[14]",
                },
            ]
        )

        self.assertTrue(expected_base_csv.equals(base_csv), "base.csv is not matching!")
        annotation_csv_file_path = os.path.join(export_path, "annotation.csv")
        self.assertTrue(os.path.exists(annotation_csv_file_path), f"Csv is not found {annotation_csv_file_path}")
        annotation_csv = pd.read_csv(annotation_csv_file_path)

        expected_annotation_csv = pd.DataFrame(
            [
                {
                    "id": 0,
                    "attributes": "[0]",
                    "type": "Polygon",
                    "name": "stickman",
                    "points": "[0, 0, 0, 50, 50, 0, 0, 0]",
                },
                {
                    "id": 1,
                    "attributes": "[0, 1]",
                    "type": "Bbox",
                    "name": "stickman",
                    "points": "[0, 0, 50, 50]",
                },
                {
                    "id": 2,
                    "attributes": "[2]",
                    "type": "Label",
                    "name": "mask",
                    "points": "[]",
                },
                {
                    "id": 3,
                    "attributes": "[3]",
                    "type": "Label",
                    "name": "scene_forest",
                    "points": "[]",
                },
                {
                    "id": 4,
                    "attributes": "[0]",
                    "type": "Bbox",
                    "name": "stickman",
                    "points": "[0, 0, 100, 100]",
                },
                {
                    "id": 5,
                    "attributes": "[4, 5]",
                    "type": "Label",
                    "name": "scene_forest",
                    "points": "[]",
                },
                {
                    "id": 6,
                    "attributes": "[0]",
                    "type": "Polygon",
                    "name": "stickman",
                    "points": "[0, 0, 0, 80, 80, 70, 60, 0]",
                },
                {
                    "id": 7,
                    "attributes": "[6]",
                    "type": "Label",
                    "name": "mask",
                    "points": "[]",
                },
                {
                    "id": 8,
                    "attributes": "[0]",
                    "type": "Bbox",
                    "name": "stickman",
                    "points": "[0, 0, 0, 50]",
                },
                {
                    "id": 9,
                    "attributes": "[0]",
                    "type": "Bbox",
                    "name": "stickman",
                    "points": "[0, 0, 180, 50]",
                },
            ]
        )
        self.assertTrue(expected_annotation_csv.equals(annotation_csv), "annotation.csv is wrong!")
        attribute_csv_file_path = os.path.join(export_path, "attribute.csv")
        self.assertTrue(os.path.exists(attribute_csv_file_path), f"csv is not found {attribute_csv_file_path}")
        attribute_csv = pd.read_csv(attribute_csv_file_path)

        expected_attribute_csv = pd.DataFrame(
            [
                {"id": 0, "name": "animation_quality", "value": "good"},
                {"id": 1, "name": "visibility", "value": "bad"},
                {"id": 2, "name": "is_running", "value": "harmful"},
                {"id": 3, "name": "animation_quality", "value": "nice"},
                {"id": 4, "name": "flip_book_animation", "value": "True"},
                {"id": 5, "name": "custom", "value": "thing"},
                {"id": 6, "name": "occluded", "value": "False"},
            ]
        )

        self.assertTrue(expected_attribute_csv.equals(attribute_csv), "attribute.csv is wrong!")
        frames_csv_file_path = os.path.join(export_path, "frames.csv")
        self.assertTrue(os.path.exists(frames_csv_file_path), f"Csv is not found {frames_csv_file_path}")
        frames_csv = pd.read_csv(frames_csv_file_path)

        expected_frames_csv = pd.DataFrame(
            [
                {"id": 0, "path": f"{relative_path}frame_0.PNG", "size": "[1920, 1080]"},
                {"id": 1, "path": f"{relative_path}frame_0__prev.PNG", "size": "[1920, 1080]"},
                {"id": 2, "path": f"{relative_path}frame_0__target.PNG", "size": "[1920, 1080]"},
                {"id": 3, "path": f"{relative_path}frame_1.PNG", "size": "[1920, 1080]"},
                {"id": 4, "path": f"{relative_path}frame_1__prev.PNG", "size": "[1920, 1080]"},
                {"id": 5, "path": f"{relative_path}frame_1__target.PNG", "size": "[1920, 1080]"},
                {"id": 6, "path": f"{relative_path}frame_2.PNG", "size": "[1920, 1080]"},
                {"id": 7, "path": f"{relative_path}frame_2__prev.PNG", "size": "[1920, 1080]"},
                {"id": 8, "path": f"{relative_path}frame_2__target.PNG", "size": "[1920, 1080]"},
                {"id": 9, "path": f"{relative_path}frame_3.PNG", "size": "[1920, 1080]"},
                {"id": 10, "path": f"{relative_path}frame_3__prev.PNG", "size": "[1920, 1080]"},
                {"id": 11, "path": f"{relative_path}frame_3__target.PNG", "size": "[1920, 1080]"},
                {"id": 12, "path": f"{relative_path}frame_4.PNG", "size": "[1920, 1080]"},
                {"id": 13, "path": f"{relative_path}frame_4__prev.PNG", "size": "[1920, 1080]"},
                {"id": 14, "path": f"{relative_path}frame_4__target.PNG", "size": "[1920, 1080]"},
            ]
        )

        self.assertTrue(expected_frames_csv.equals(frames_csv), "frames.csv is wrong!")
