from __future__ import annotations

import os
import unittest

import cv2
import datumaro as dm
import numpy as np
import yaml
from cv2 import COLOR_BGR2GRAY
from cv2 import cvtColor
from datumaro.components.annotation import (
    AnnotationType,
)

from datumaid import DatasetManager
from test import utils


class TestFillColor(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.sep + "tmp" + os.sep + "test_fill_color" + os.sep
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.frame_0_path = self.test_dir + "frame0.jpg"
        self.frame_1_path = self.test_dir + "frame1.jpg"

        frame_0 = utils.generate_test_image()
        frame_1 = utils.generate_test_image(channel=1)

        cv2.imwrite(self.frame_0_path, frame_0)
        cv2.imwrite(self.frame_1_path, frame_1)

        self.frame_0 = cv2.imread(self.frame_0_path)
        frame_1 = cv2.imread(self.frame_1_path)
        self.frame_1 = cvtColor(frame_1, COLOR_BGR2GRAY)

    def tearDown(self):
        if not os.path.exists(self.test_dir):
            os.removedirs(self.test_dir)

    def test_fill_bboxes(self):
        test_config = yaml.safe_load(
            """
datasets:
- dataset:
  item_manipulations:
    - item:
      bboxes:
        - bbox: mask
      exec: image_module.fill_color(item, item_yml, color=(125,125,125))
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    image=self.frame_0_path,
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(50, 50, 50, 50, id=1, label=1),  # fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    image=self.frame_1_path,
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=2, label=0),  # no fill
                        dm.Bbox(50, 50, 50, 50, id=3, label=1),  # fill
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman", "mask"],
                    ["animation_quality", "occluded"],
                )
            },
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_org)

        frame_0_out = self.frame_0.copy()
        cv2.rectangle(frame_0_out, (50, 50), (100, 100), (125, 125, 125), -1)  # for bbox id 1
        frame_0_org = cv2.imread(self.frame_0_path)
        utils.compare_images(self, frame_0_out, frame_0_org, 35)
        frame_1_out = self.frame_1.copy()
        cv2.rectangle(frame_1_out, (50, 50), (100, 100), (125, 125, 125), -1)  # for bbox id 4
        frame_1_org = cv2.imread(self.frame_1_path)
        frame_1_org = cvtColor(frame_1_org, COLOR_BGR2GRAY)
        utils.compare_images(self, frame_1_out, frame_1_org, 20)

    def test_fill_bboxes_with_attrs(self):
        test_config = yaml.safe_load(
            """
datasets:
- dataset:
  item_manipulations:
    - item:
      bboxes:
        - bbox: mask
        - bbox: stickman
          attr:
            animation_quality: bad
      exec: image_module.fill_color(item, item_yml, color=(125,125,125))
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    image=self.frame_0_path,
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(
                            50,
                            50,
                            50,
                            50,
                            id=1,
                            label=0,
                            attributes={"animation_quality": "bad"},
                        ),  # fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),  # no fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"occluded": True}),  # no fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    image=self.frame_1_path,
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                        dm.Bbox(
                            50,
                            50,
                            50,
                            50,
                            id=4,
                            label=1,
                            attributes={"occluded": True},
                        ),  # fill
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman", "mask"],
                    ["animation_quality", "occluded"],
                )
            },
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_org)

        frame_0_out = self.frame_0.copy()
        cv2.rectangle(frame_0_out, (50, 50), (100, 100), (125, 125, 125), -1)  # for bbox id 1
        frame_0_org = cv2.imread(self.frame_0_path)
        utils.compare_images(self, frame_0_out, frame_0_org, 35)
        frame_1_out = self.frame_1.copy()
        cv2.rectangle(frame_1_out, (0, 0), (50, 50), (125, 125, 125), -1)  # for bbox id 3
        cv2.rectangle(frame_1_out, (50, 50), (100, 100), (125, 125, 125), -1)  # for bbox id 4
        frame_1_org = cv2.imread(self.frame_1_path)
        frame_1_org = cvtColor(frame_1_org, COLOR_BGR2GRAY)
        utils.compare_images(self, frame_1_out, frame_1_org, 20)

    def test_fill_polygons(self):
        test_config = yaml.safe_load(
            """
datasets:
- dataset:
  item_manipulations:
    - item:
      polygons:
        - polygon: stickman
          attr:
            animation_quality: bad
            occluded: True
      exec: image_module.fill_color(item, item_yml, color=(125,125,125))
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    image=self.frame_0_path,
                    annotations=[
                        dm.Polygon([0, 0, 100, 50, 0, 100], id=0, label=0),  # no fill
                        dm.Polygon(
                            [0, 0, 100, 50, 0, 100],
                            id=1,
                            label=0,
                            attributes={"animation_quality": "not bad"},
                        ),  # no fill
                        dm.Polygon(
                            [0, 0, 100, 50, 0, 100],
                            id=2,
                            label=0,
                            attributes={"animation_quality": "bad"},
                        ),  # no fill
                        dm.Polygon(
                            [100, 100, 200, 150, 100, 200, 0, 0],
                            id=3,
                            label=0,
                            attributes={"animation_quality": "bad", "occluded": True},
                        ),  # fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    image=self.frame_1_path,
                    annotations=[
                        dm.Polygon([0, 0, 100, 50, 0, 100], id=2, label=0),  # no fill
                        dm.Polygon(
                            [0, 0, 100, 50, 0, 100],
                            id=4,
                            label=0,
                            attributes={"occluded": True},
                        ),  # no fill
                        dm.Polygon(
                            [0, 0, 100, 50, 0, 100],
                            id=5,
                            label=0,
                            attributes={"occluded": False},
                        ),  # no fill
                        dm.Polygon(
                            [100, 100, 200, 150, 100, 200],
                            id=6,
                            label=0,
                            attributes={"animation_quality": "bad", "occluded": True},
                        ),  # fill
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman", "mask"],
                    ["animation_quality", "occluded"],
                )
            },
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_org)

        frame_0_out = self.frame_0.copy()
        cv2.fillPoly(
            frame_0_out,
            pts=np.int32([[[100, 100], [200, 150], [100, 200], [0, 0]]]),
            color=(125, 125, 125),
        )
        frame_0_org = cv2.imread(self.frame_0_path)
        utils.compare_images(self, frame_0_out, frame_0_org, 35)
        frame_1_out = self.frame_1.copy()
        cv2.fillPoly(
            frame_1_out,
            pts=np.int32([[[100, 100], [200, 150], [100, 200]]]),
            color=(125, 125, 125),
        )
        frame_1_org = cv2.imread(self.frame_1_path)
        frame_1_org = cvtColor(frame_1_org, COLOR_BGR2GRAY)
        utils.compare_images(self, frame_1_out, frame_1_org, 20)

    def test_fill_polygons_with_attrs(self):
        test_config = yaml.safe_load(
            """
datasets:
- dataset:
  item_manipulations:
    - item:
      bboxes:
        - bbox: mask
        - bbox: stickman
          attr:
            animation_quality: bad
      exec: image_module.fill_color(item, item_yml, color=(125,125,125))
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    image=self.frame_0_path,
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(
                            50,
                            50,
                            50,
                            50,
                            id=1,
                            label=0,
                            attributes={"animation_quality": "bad"},
                        ),  # fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),  # no fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"occluded": True}),  # no fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    image=self.frame_1_path,
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                        dm.Bbox(
                            50,
                            50,
                            50,
                            50,
                            id=4,
                            label=1,
                            attributes={"occluded": True},
                        ),  # fill
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman", "mask"],
                    ["animation_quality", "occluded"],
                )
            },
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_org)

        frame_0_out = self.frame_0.copy()
        cv2.rectangle(frame_0_out, (50, 50), (100, 100), (125, 125, 125), -1)  # for bbox id 1
        frame_0_org = cv2.imread(self.frame_0_path)
        utils.compare_images(self, frame_0_out, frame_0_org, 35)
        frame_1_out = self.frame_1.copy()
        cv2.rectangle(frame_1_out, (0, 0), (50, 50), (125, 125, 125), -1)  # for bbox id 3
        cv2.rectangle(frame_1_out, (50, 50), (100, 100), (125, 125, 125), -1)  # for bbox id 4
        frame_1_org = cv2.imread(self.frame_1_path)
        frame_1_org = cvtColor(frame_1_org, COLOR_BGR2GRAY)
        utils.compare_images(self, frame_1_out, frame_1_org, 20)


class TestBitwiseNot(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.sep + "tmp" + os.sep + "test_bitwise_not" + os.sep
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.frame_0_path = self.test_dir + "frame0.jpg"
        self.frame_1_path = self.test_dir + "frame1.jpg"
        self.frame_2_path = self.test_dir + "frame2.jpg"

        frame_0 = utils.generate_test_image()
        frame_1 = utils.generate_test_image(channel=1)
        frame_2 = utils.generate_test_image()

        cv2.imwrite(self.frame_0_path, frame_0)
        cv2.imwrite(self.frame_1_path, frame_1)
        cv2.imwrite(self.frame_2_path, frame_2)

        self.frame_0 = cv2.imread(self.frame_0_path)
        frame_1 = cv2.imread(self.frame_1_path)
        self.frame_1 = cvtColor(frame_1, COLOR_BGR2GRAY)
        self.frame_2 = cv2.imread(self.frame_0_path)

    def tearDown(self):
        if not os.path.exists(self.test_dir):
            os.removedirs(self.test_dir)

    def test_bitwise_not_image_if_label_exists(self):
        test_config = yaml.safe_load(
            """
datasets:
- dataset:
  item_manipulations:
    - item:
      labels:
        - label: camera_ir_wh
      exec: image_module.bitwise_not(item, item_yml)
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    image=self.frame_0_path,
                    annotations=[
                        dm.Label(id=0, label=0),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    image=self.frame_1_path,
                    annotations=[
                        dm.Label(id=0, label=0),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    image=self.frame_2_path,
                    annotations=[
                        dm.Label(id=0, label=1),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["camera_ir_wh", "camera_ir_bh"],
                    ["animation_quality", "occluded"],
                )
            },
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_org)

        frame_0_out = self.frame_0.copy()
        frame_0_out = cv2.bitwise_not(frame_0_out)
        frame_0_org = cv2.imread(self.frame_0_path)
        utils.compare_images(self, frame_0_out, frame_0_org, 30)

        frame_1_out = self.frame_1.copy()
        frame_1_out = cv2.bitwise_not(frame_1_out)
        frame_1_org = cv2.imread(self.frame_1_path)
        frame_1_org = cvtColor(frame_1_org, COLOR_BGR2GRAY)
        utils.compare_images(self, frame_1_out, frame_1_org, 20)

        frame_2_out = self.frame_2.copy()
        frame_2_org = cv2.imread(self.frame_2_path)
        utils.compare_images(self, frame_2_out, frame_2_org, 20)

    def test_bitwise_not_image_if_annotation_exists(self):
        test_config = yaml.safe_load(
            """
datasets:
- dataset:
  item_manipulations:
    - item:
      bboxes:
        - bbox: stickman
          attr:
            occluded: True
      polygons:
        - polygon: stickman
          attr:
            occluded: True
      exec: image_module.bitwise_not(item, item_yml)
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    image=self.frame_0_path,
                    annotations=[
                        dm.Polygon(
                            [0, 0, 50, 50, 100, 100, 50, 50],
                            id=0,
                            label=0,
                            attributes={"occluded": True},
                        ),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    image=self.frame_1_path,
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            50,
                            50,
                            id=0,
                            label=0,
                            attributes={"occluded": True, "animation_quality": "bad"},
                        ),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    image=self.frame_2_path,
                    annotations=[
                        dm.Bbox(
                            0,
                            0,
                            50,
                            50,
                            id=0,
                            label=0,
                            attributes={"occluded": False, "animation_quality": "bad"},
                        ),
                        dm.Bbox(0, 0, 50, 50, id=0, label=1),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman", "animal"],
                    ["animation_quality", "occluded"],
                )
            },
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_org)

        frame_0_out = self.frame_0.copy()
        frame_0_out = cv2.bitwise_not(frame_0_out)
        frame_0_org = cv2.imread(self.frame_0_path)
        utils.compare_images(self, frame_0_out, frame_0_org, 30)

        frame_1_out = self.frame_1.copy()
        frame_1_out = cv2.bitwise_not(frame_1_out)
        frame_1_org = cv2.imread(self.frame_1_path)
        frame_1_org = cvtColor(frame_1_org, COLOR_BGR2GRAY)
        utils.compare_images(self, frame_1_out, frame_1_org, 20)

        frame_2_out = self.frame_2.copy()
        frame_2_org = cv2.imread(self.frame_2_path)
        utils.compare_images(self, frame_2_out, frame_2_org, 20)


class TestRemoveAnnotation(unittest.TestCase):
    def test_remove_annotations(self):
        test_config = yaml.safe_load(
            """
datasets:
- dataset:
  item_manipulations:
    - item:
      bboxes:
        - bbox: stickman
          attr:
            animation_quality: bad
      polygons:
        - polygon: mask
          attr:
            occluded: True
      labels:
        - label: camera_ir_wh
      exec: selection_module.remove_annotation(item, item_yml)
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(0, 0, 50, 50, id=0, label=0, attributes={"animation_quality": "bad"}),
                        dm.Bbox(0, 0, 50, 50, id=0, label=0, attributes={"animation_quality": "not bad"}),
                        dm.Label(id=0, label=1),
                        dm.Label(id=0, label=2),
                        dm.Label(id=0, label=2, attributes={"animation_quality": "not bad"}),
                        dm.Polygon([0, 0, 100, 100, 200, 200, 50, 50], label=1),
                        dm.Polygon(
                            [0, 0, 100, 100, 200, 200, 50, 50],
                            label=1,
                            attributes={"occluded": True},
                        ),
                        dm.Polygon(
                            [0, 0, 100, 100, 200, 200, 50, 50],
                            label=0,
                            attributes={"animation_quality": "bad"},
                        ),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[
                        dm.Label(id=0, label=2),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman", "mask", "camera_ir_wh"],
                    ["animation_quality", "occluded"],
                )
            },
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()

        expected = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(0, 0, 50, 50, id=0, label=0, attributes={"animation_quality": "not bad"}),
                        dm.Label(id=0, label=1),
                        dm.Polygon([0, 0, 100, 100, 200, 200, 50, 50], label=1),
                        dm.Polygon(
                            [0, 0, 100, 100, 200, 200, 50, 50],
                            label=0,
                            attributes={"animation_quality": "bad"},
                        ),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman", "mask", "camera_ir_wh"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, expected)
