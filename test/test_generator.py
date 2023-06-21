from __future__ import annotations

import os
import random
import shutil
import unittest
from copy import deepcopy

import datumaro as dm
import yaml
from datumaro.components.annotation import (
    AnnotationType,
)

from datumaid import DatasetManager
from test import utils


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.sep + "tmp" + os.sep + "test_generator"
        os.makedirs(self.test_dir, exist_ok=True)

        self.video_name = utils.create_sample_video(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def check_created_lists(self, exp_original, exp_pollute, original_items, polluted_items):
        def check_item_ids(items, ids):
            for item in items:
                self.assertTrue(item.id in ids)
                ids.remove(item.id)

            self.assertEqual(len(ids), 0)

        check_item_ids(original_items, exp_original)

        for polluted_items, expected_ids in zip(
            polluted_items,
            exp_pollute,
        ):
            check_item_ids(polluted_items, expected_ids)

    def check_created_lists_not_duplicate(self, items, dataset):
        item_pairs = []
        for item in items:
            item_pairs.append([item])

        item_ids = [key[0].id for key in item_pairs]

        for item in dataset:
            if item.id in item_ids:
                item_pairs[item_ids.index(item.id)].append(item)

        # Change first item id and check if pairs id is changed
        for idx, (item, item_pair) in enumerate(item_pairs):
            item.id = random.randint(0, 25000)
            self.assertEqual(item_pair.id, item.id)

    def test_dataset_item_deepcopy(self):
        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[
                        dm.Label(id=0, label=0),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        item = datum_dataset_org.get("frame_000000")
        item_copy = deepcopy(item)
        item_copy.id = "frame_000001"
        item_copy.annotations[0].label = 3

        self.assertNotEqual(item.id, item_copy.id)
        self.assertNotEqual(item.annotations[0].label, item_copy.annotations[0].label)

    def test_pollute_annotations(self):
        test_config = yaml.safe_load(
            f"""
datasets:
- dataset:
  generator:
    pollute_back: 3
    pollute_step: 1
    video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 5)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

        # Compare images
        for item in d_manager.config_yml["datasets"][0]["datum_datasets"][0]:
            self.assertTrue(utils.compare_sample_video_frame_ids(item.image.data, item.id))

    def test_pollute_step(self):
        test_config = yaml.safe_load(
            f"""
datasets:
- dataset:
  generator:
    pollute_back: 3
    pollute_step: 2
    video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),  # no fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 4)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),  # no fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),  # no fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

        # Compare images
        for item in d_manager.config_yml["datasets"][0]["datum_datasets"][0]:
            self.assertTrue(utils.compare_sample_video_frame_ids(item.image.data, item.id))

    def test_pollute_back(self):
        test_config = yaml.safe_load(
            f"""
datasets:
- dataset:
  generator:
    pollute_back: 4
    pollute_step: 1
    video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000010",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 9)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),  # no fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),  # no fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),  # no fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),  # no fill
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),  # no fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000009",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000010",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),  # fill
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

        # Compare images
        for item in d_manager.config_yml["datasets"][0]["datum_datasets"][0]:
            self.assertTrue(utils.compare_sample_video_frame_ids(item.image.data, item.id))

    def test_pollute_overlap_with_original_labels(self):
        test_config = yaml.safe_load(
            f"""
datasets:
- dataset:
  generator:
    pollute_back: 4
    pollute_step: 4
    video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[dm.Bbox(0, 0, 50, 50, id=3, label=1)],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

        # Compare images
        for item in d_manager.config_yml["datasets"][0]["datum_datasets"][0]:
            self.assertTrue(utils.compare_sample_video_frame_ids(item.image.data, item.id))

    def test_pollute_beyond(self):
        test_config = yaml.safe_load(
            f"""
datasets:
- dataset:
  generator:
    pollute_back: 5
    pollute_step: 5
    video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=1),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 4)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=0, label=0),
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[dm.Bbox(0, 0, 50, 50, id=3, label=1)],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

        # Compare images
        for item in d_manager.config_yml["datasets"][0]["datum_datasets"][0]:
            self.assertTrue(utils.compare_sample_video_frame_ids(item.image.data, item.id))

    def test_pollute_overlap_with_polluted_labels(self):
        test_config = yaml.safe_load(
            f"""
datasets:
- dataset:
  generator:
    pollute_back: 3
    pollute_step: 1
    video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000010",
                    annotations=[
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=0),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 6)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000009",
                    annotations=[
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000010",
                    annotations=[
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000011",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=0),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=0),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

        # Compare images
        for item in d_manager.config_yml["datasets"][0]["datum_datasets"][0]:
            self.assertTrue(utils.compare_sample_video_frame_ids(item.image.data, item.id))

    def test_pollute_created_lists(self):
        test_config = yaml.safe_load(
            f"""
datasets:
- dataset:
  generator:
    pollute_back: 3
    pollute_step: 1
    video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000010",
                    annotations=[
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=0),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 6)

        expected_original_ids = ["frame_000010", "frame_000012"]
        expected_polluted_ids = [["frame_000009", "frame_000008", "frame_000007"], ["frame_000011", "frame_000010", "frame_000009"]]

        self.check_created_lists(
            expected_original_ids,
            expected_polluted_ids,
            d_manager.config_yml["original_anns"],
            d_manager.config_yml["polluted_anns"],
        )

    def test_pollute_traversal_order(self):
        test_config = yaml.safe_load(
            f"""
datasets:
- dataset:
  generator:
    pollute_back: 3
    pollute_step: 1
    video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000010",
                    annotations=[
                        dm.Bbox(100, 100, 50, 50, id=2, label=0, attributes={"animation_quality": "not bad"}),
                    ],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[
                        dm.Bbox(0, 0, 50, 50, id=3, label=0),
                    ],
                ),
            ],
            categories=utils.get_test_categories(),
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 6)

        expected_traversal_order = [
            "frame_000007",
            "frame_000008",
            "frame_000009",
            "frame_000010",
            "frame_000011",
            "frame_000012",
        ]
        for item, expected_id in zip(d_manager.config_yml["datasets"][0]["datum_datasets"][0], expected_traversal_order):
            self.assertEqual(item.id, expected_id)

    def test_select_before_generator(self):
        test_config = yaml.safe_load(
            f"""
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: stickman
            attr:
              animation_quality: good

        exec: selection_module.any_of_annotations(item, select_yml)

    generator:
        pollute_back: 2
        pollute_step: 1
        video_path: {self.video_name}
"""
        )
        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Label(id=1, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[dm.Label(id=2, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000012",
                    annotations=[dm.Label(id=1, label=1)],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman"],
                    ["animation_quality"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 4)

        d_manager.process()
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Label(id=2, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Label(id=2, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[dm.Label(id=2, label=1, attributes={"animation_quality": "good"})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman"],
                    ["animation_quality"],
                )
            },
        )

        expected_original_ids = ["frame_000008"]
        expected_polluted_ids = [["frame_000007", "frame_000006"]]

        self.check_created_lists(
            expected_original_ids,
            expected_polluted_ids,
            d_manager.config_yml["original_anns"],
            d_manager.config_yml["polluted_anns"],
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

        # Compare images
        for item in d_manager.config_yml["datasets"][0]["datum_datasets"][0]:
            self.assertTrue(utils.compare_sample_video_frame_ids(item.image.data, item.id))

    def test_remove_annotations_after_generator(self):
        test_config = yaml.safe_load(
            f"""
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

  generator:
      pollute_back: 2
      pollute_step: 2
      video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000002",
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
                    id="frame_000005",
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
                    id="frame_000002",
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
                    id="frame_000003",
                    annotations=[],
                ),
                dm.DatasetItem(
                    id="frame_000005",
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

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 4)
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, expected)

        # Compare images
        for item in d_manager.config_yml["datasets"][0]["datum_datasets"][0]:
            self.assertTrue(utils.compare_sample_video_frame_ids(item.image.data, item.id))

        expected_original_ids = ["frame_000002", "frame_000005"]
        expected_polluted_ids = [["frame_000000"], ["frame_000003"]]

        self.check_created_lists(
            expected_original_ids,
            expected_polluted_ids,
            d_manager.config_yml["original_anns"],
            d_manager.config_yml["polluted_anns"],
        )

    def test_remap_labels_delete(self):
        test_config = yaml.safe_load(
            f"""
datasets:
    - dataset:
      remap_labels:
        default: delete
        mapping:
          stickman: stickman
          key_point_animated:

      generator:
        pollute_back: 2
        pollute_step: 2
        video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1)],
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
                    annotations=[dm.Label(id=2, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000005",
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
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 4)

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, expected)

        # Compare images
        for item in d_manager.config_yml["datasets"][0]["datum_datasets"][0]:
            self.assertTrue(utils.compare_sample_video_frame_ids(item.image.data, item.id))

        expected_original_ids = ["frame_000002", "frame_000007"]
        expected_polluted_ids = [["frame_000000"], ["frame_000005"]]

        self.check_created_lists(
            expected_original_ids,
            expected_polluted_ids,
            d_manager.config_yml["original_anns"],
            d_manager.config_yml["polluted_anns"],
        )

    def test_split_train_test(self):
        test_config = yaml.safe_load(
            f"""
datasets:
  - dataset:
    split:
      subset:
        train: 0.8
        val: 0.2

    generator:
      pollute_back: 2
      pollute_step: 2
      video_path: {self.video_name}
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(id="frame_000000"),
                dm.DatasetItem(id="frame_000004", annotations=[dm.Label(id=1, label=0)]),
                dm.DatasetItem(id="frame_000010"),
            ],
        )

        expected = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(id="frame_000000", subset="train"),
                dm.DatasetItem(id="frame_000002", annotations=[dm.Label(id=1, label=0)], subset="train"),
                dm.DatasetItem(id="frame_000004", annotations=[dm.Label(id=1, label=0)], subset="train"),
                dm.DatasetItem(id="frame_000008", subset="val"),
                dm.DatasetItem(id="frame_000010", subset="train"),
            ],
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)

        d_manager.process()
        self.assertEqual(4, len(d_manager.config_yml["datasets"][0]["datum_datasets"][0].get_subset("train")))
        self.assertEqual(1, len(d_manager.config_yml["datasets"][0]["datum_datasets"][0].get_subset("val")))
        utils.compare_datasets(self, test_config["datasets"][0]["datum_datasets"][0], expected)

        expected_original_ids = ["frame_000000", "frame_000004", "frame_000010"]
        expected_polluted_ids = [[], ["frame_000002"], ["frame_000008"]]

        self.check_created_lists(
            expected_original_ids,
            expected_polluted_ids,
            d_manager.config_yml["original_anns"],
            d_manager.config_yml["polluted_anns"],
        )

    def test_split_duplicate_check(self):
        test_config = yaml.safe_load(
            f"""
datasets:
  - dataset:
    split:
      subset:
        train: 0.8
        val: 0.2

    generator:
      pollute_back: 2
      pollute_step: 2
      video_path: {self.video_name}
project_labels: [stickman]
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(id="frame_000000"),
                dm.DatasetItem(id="frame_000004", annotations=[dm.Label(id=1, label=0)]),
                dm.DatasetItem(id="frame_000010"),
            ],
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)

        d_manager.process()

        list_items = d_manager.config_yml["original_anns"]
        list_items.append(d_manager.config_yml["polluted_anns"][1][0])

        self.check_created_lists_not_duplicate(list_items, d_manager.config_yml["datasets"][0]["datum_datasets"][0])
        self.check_created_lists_not_duplicate(list_items, d_manager.config_yml["processed_dataset"])

    def test_remap_dublicate_check(self):
        test_config = yaml.safe_load(
            f"""
datasets:
    - dataset:
      remap_labels:
        default: delete
        mapping:
          stickman: stickman
          key_point_animated:

      generator:
        pollute_back: 2
        pollute_step: 2
        video_path: {self.video_name}
project_labels: [stickman]
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1)],
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

        list_items = d_manager.config_yml["original_anns"]
        list_items.append(d_manager.config_yml["polluted_anns"][1][0])

        self.check_created_lists_not_duplicate(list_items, d_manager.config_yml["datasets"][0]["datum_datasets"][0])
        self.check_created_lists_not_duplicate(list_items, d_manager.config_yml["processed_dataset"])

    def test_transform_labels_duplicate_check(self):
        test_config = yaml.safe_load(
            f"""
datasets:
  - dataset:

    generator:
      pollute_back: 2
      pollute_step: 1
      video_path: {self.video_name}
project_labels: [stickman]
"""
        )

        datum_dataset = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=1, attributes={"occluded": True})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman", "scene_flat"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset]
        d_manager = DatasetManager(config_yml=test_config)
        d_manager.process()

        list_items = d_manager.config_yml["original_anns"]
        list_items.append(d_manager.config_yml["polluted_anns"][1][0])

        self.check_created_lists_not_duplicate(list_items, d_manager.config_yml["datasets"][0]["datum_datasets"][0])
        self.check_created_lists_not_duplicate(list_items, d_manager.config_yml["processed_dataset"])

    def test_multiple_dataset_reindex_duplicate_check(self):
        test_config = yaml.safe_load(
            f"""
datasets:
  - dataset:

    generator:
      pollute_back: 2
      pollute_step: 1
      video_path: {self.video_name}

project_labels: [stickman]
"""
        )

        datum_dataset1 = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=1, attributes={"occluded": True})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman", "scene_flat"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        datum_dataset2 = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=0, label=0, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=1, attributes={"occluded": True})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman", "scene_flat"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset1, datum_dataset2]

        d_manager = DatasetManager(config_yml=test_config)
        d_manager.process()

        list_items = d_manager.config_yml["original_anns"]
        list_items.append(d_manager.config_yml["polluted_anns"][1][0])

        self.check_created_lists_not_duplicate(list_items, d_manager.config_yml["processed_dataset"])
