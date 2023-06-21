from __future__ import annotations

import unittest

import datumaro as dm
import yaml
from datumaro.components.annotation import AnnotationType

from datumaid import DatasetManager
from test import utils


class TestTransform(unittest.TestCase):
    def test_split_train_test_keep_defaults(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    split:
      keep_defaults: True
      subset:
        train: 0.8
        val: 0.2
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(id=0, annotations=[dm.Label(id=0, label=0)], subset="val"),
                dm.DatasetItem(id=1, annotations=[dm.Label(id=1, label=0)], subset="default"),
                dm.DatasetItem(id="ab", subset="a"),
                dm.DatasetItem(id=2, subset="train"),
                dm.DatasetItem(id="a", subset="test"),
                dm.DatasetItem(id=3),
                dm.DatasetItem(id=4),
                dm.DatasetItem(id=5, subset="train"),
                dm.DatasetItem(id="as", subset="a"),
                dm.DatasetItem(id=6),
            ],
        )

        expected = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(id=0, annotations=[dm.Label(id=0, label=0)], subset="val"),
                dm.DatasetItem(id=1, annotations=[dm.Label(id=1, label=0)], subset="train"),
                dm.DatasetItem(id="ab", subset="train"),
                dm.DatasetItem(id=2, subset="train"),
                dm.DatasetItem(id="a", subset="val"),
                dm.DatasetItem(id=3, subset="train"),
                dm.DatasetItem(id=4, subset="train"),
                dm.DatasetItem(id=5, subset="train"),
                dm.DatasetItem(id="as", subset="train"),
                dm.DatasetItem(id=6, subset="val"),
            ],
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)

        d_manager.process()

        self.assertEqual(7, len(datum_dataset_org.get_subset("train")))
        self.assertEqual(3, len(datum_dataset_org.get_subset("val")))
        utils.compare_datasets(self, datum_dataset_org, expected)

    def test_split_train_test(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    split:
      subset:
        train: 0.8
        val: 0.2
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(id=0, annotations=[dm.Label(id=0, label=0)], subset="val"),
                dm.DatasetItem(id=1, annotations=[dm.Label(id=1, label=0)], subset="default"),
                dm.DatasetItem(id="ab", subset="a"),
                dm.DatasetItem(id=2, subset="train"),
                dm.DatasetItem(id="a", subset="test"),
                dm.DatasetItem(id=3),
                dm.DatasetItem(id=4),
                dm.DatasetItem(id=5, subset="train"),
                dm.DatasetItem(id="as", subset="a"),
                dm.DatasetItem(id=6),
            ],
        )

        expected = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(id=0, annotations=[dm.Label(id=0, label=0)], subset="train"),
                dm.DatasetItem(id=1, annotations=[dm.Label(id=1, label=0)], subset="train"),
                dm.DatasetItem(id="ab", subset="train"),
                dm.DatasetItem(id=2, subset="train"),
                dm.DatasetItem(id="a", subset="val"),
                dm.DatasetItem(id=3, subset="train"),
                dm.DatasetItem(id=4, subset="train"),
                dm.DatasetItem(id=5, subset="train"),
                dm.DatasetItem(id="as", subset="train"),
                dm.DatasetItem(id=6, subset="val"),
            ],
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)

        d_manager.process()

        self.assertEqual(8, len(datum_dataset_org.get_subset("train")))
        self.assertEqual(2, len(datum_dataset_org.get_subset("val")))
        utils.compare_datasets(self, datum_dataset_org, expected)

    def test_remap_labels_delete(self):
        test_config = yaml.safe_load(
            """
datasets:
    - dataset:
      remap_labels:
        default: delete
        mapping:
          stickman: stickman
          key_point_animated:
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=1, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=3, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=4, label=1, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=5, label=1, attributes={"animation_quality": "bad"})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=5, label=2, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=6, label=2, attributes={"animation_quality": "bad"})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman", "scene_flat"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        d_manager.process()

        expected = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                ),
                dm.DatasetItem(
                    id="frame_000001",
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=3, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=4, label=0, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=5, label=0, attributes={"animation_quality": "bad"})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                ),
                dm.DatasetItem(
                    id="frame_000007",
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        utils.compare_datasets(self, datum_dataset_org, expected)

    def test_remap_labels_keep(self):
        test_config = yaml.safe_load(
            """
datasets:
    - dataset:
      remap_labels:
        default: keep
        mapping:
          stickman: stickman
          key_point_animated:
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=1, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=3, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=4, label=1, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=5, label=1, attributes={"animation_quality": "bad"})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=5, label=2, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=6, label=2, attributes={"animation_quality": "bad"})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman", "scene_flat"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        d_manager.process()

        expected = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                ),
                dm.DatasetItem(
                    id="frame_000001",
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=3, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=4, label=0, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=5, label=0, attributes={"animation_quality": "bad"})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=5, label=1, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=6, label=1, attributes={"animation_quality": "bad"})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman", "scene_flat"],
                    ["animation_quality", "occluded"],
                )
            },
        )
        utils.compare_datasets(self, datum_dataset_org, expected)

    def test_remap_labels_default_keep(self):
        test_config = yaml.safe_load(
            """
datasets:
    - dataset:
      remap_labels:
        mapping:
          stickman: person
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=1, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1)],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        d_manager.process()

        expected = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=1, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1)],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "person"],
                    ["animation_quality", "occluded"],
                )
            },
        )
        utils.compare_datasets(self, datum_dataset_org, expected)

    def test_reindex_dataset_after_process(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
project_labels: [stickman]
"""
        )

        datum_dataset_1 = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=0, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=0, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=1, attributes={"occluded": True})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=0, label=1, attributes={"animation_quality": "bad"})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=2, attributes={"occluded": False})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=0, label=2, attributes={"animation_quality": "bad"})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman", "scene_flat"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        datum_dataset_2 = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=0, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=0, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=1, attributes={"umbrella": True})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=0, label=1, attributes={"umbrella": True})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=2, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=0, label=2, attributes={"animation_quality": "bad"})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman", "mask"],
                    ["animation_quality", "umbrella"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_1, datum_dataset_2]
        d_manager = DatasetManager(config_yml=test_config)
        d_manager.process()

        expected = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id=0,
                ),
                dm.DatasetItem(
                    id=1,
                ),
                dm.DatasetItem(
                    id=2,
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id=3,
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id=4,
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=0, attributes={"occluded": True})],
                ),
                dm.DatasetItem(
                    id=5,
                    annotations=[
                        dm.Polygon(
                            [0, 1, 2, 3, 4, 5],
                            id=0,
                            label=0,
                            attributes={"animation_quality": "bad"},
                        )
                    ],
                ),
                dm.DatasetItem(
                    id=6,
                ),
                dm.DatasetItem(
                    id=7,
                ),
                dm.DatasetItem(
                    id=8,
                ),
                dm.DatasetItem(
                    id=9,
                ),
                dm.DatasetItem(
                    id=10,
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id=11,
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id=12,
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=0, attributes={"umbrella": True})],
                ),
                dm.DatasetItem(
                    id=13,
                    annotations=[
                        dm.Polygon(
                            [0, 1, 2, 3, 4, 5],
                            id=0,
                            label=0,
                            attributes={"umbrella": True},
                        )
                    ],
                ),
                dm.DatasetItem(
                    id=14,
                ),
                dm.DatasetItem(
                    id=15,
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman"],
                    ["animation_quality", "occluded", "umbrella"],
                )
            },
        )
        utils.compare_datasets(self, d_manager.config_yml["processed_dataset"], expected)

    def test_reindex_all_datasets_after_process(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
  - dataset:
project_labels: [stickman]
"""
        )

        datum_dataset_1 = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=0, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=0, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=1, attributes={"occluded": True})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=0, label=1, attributes={"animation_quality": "bad"})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=2, attributes={"occluded": False})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=0, label=2, attributes={"animation_quality": "bad"})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman", "scene_flat"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        datum_dataset_2 = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=0, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=0, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=1, attributes={"umbrella": True})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=0, label=1, attributes={"umbrella": True})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=2, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Polygon([0, 1, 2, 3, 4, 5], id=0, label=2, attributes={"animation_quality": "bad"})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman", "mask"],
                    ["animation_quality", "umbrella"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_1]
        test_config["datasets"][1]["datum_datasets"] = [datum_dataset_2]
        d_manager = DatasetManager(config_yml=test_config)
        d_manager.process()

        expected = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id=0,
                ),
                dm.DatasetItem(
                    id=1,
                ),
                dm.DatasetItem(
                    id=2,
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id=3,
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id=4,
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=0, attributes={"occluded": True})],
                ),
                dm.DatasetItem(
                    id=5,
                    annotations=[
                        dm.Polygon(
                            [0, 1, 2, 3, 4, 5],
                            id=0,
                            label=0,
                            attributes={"animation_quality": "bad"},
                        )
                    ],
                ),
                dm.DatasetItem(
                    id=6,
                ),
                dm.DatasetItem(
                    id=7,
                ),
                dm.DatasetItem(
                    id=8,
                ),
                dm.DatasetItem(
                    id=9,
                ),
                dm.DatasetItem(
                    id=10,
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id=11,
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id=12,
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=0, attributes={"umbrella": True})],
                ),
                dm.DatasetItem(
                    id=13,
                    annotations=[
                        dm.Polygon(
                            [0, 1, 2, 3, 4, 5],
                            id=0,
                            label=0,
                            attributes={"umbrella": True},
                        )
                    ],
                ),
                dm.DatasetItem(
                    id=14,
                ),
                dm.DatasetItem(
                    id=15,
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["stickman"],
                    ["animation_quality", "occluded", "umbrella"],
                )
            },
        )
        utils.compare_datasets(self, d_manager.config_yml["processed_dataset"], expected)
