from __future__ import annotations

import copy
import unittest

import datumaro as dm
import yaml
from datumaro.components.annotation import AnnotationType
from datumaro.components.annotation import LabelCategories

from datumaid import DatasetManager
from test import utils


class TestAnyOfAnnotations(unittest.TestCase):
    def test_select_label(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: key_point_animated

        exec: selection_module.any_of_annotations(item, select_yml)
"""
        )

        def test_categories():
            return {AnnotationType.label: LabelCategories.from_iterable(["key_point_animated", "key_point_animated_2"])}

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=1, label=1)],
                ),
            ],
            categories=test_categories(),
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 1)
        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                )
            ],
            categories=test_categories(),
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    def test_select_label_from_multi_datasets(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: key_point_animated

        exec: selection_module.any_of_annotations(item, select_yml)

  - dataset:
    selects:
      - select:
        labels:
          - label: key_point_animated

        exec: selection_module.any_of_annotations(item, select_yml)
"""
        )

        def test_categories():
            return {AnnotationType.label: LabelCategories.from_iterable(["key_point_animated", "key_point_animated_2"])}

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=1, label=1)],
                ),
            ],
            categories=test_categories(),
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        test_config["datasets"][1]["datum_datasets"] = [copy.deepcopy(datum_dataset_org)]

        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 2)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)
        self.assertEqual(len(d_manager.config_yml["datasets"][1]["datum_datasets"][0]), 2)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][1]["datum_datasets"][0]), 1)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                )
            ],
            categories=test_categories(),
        )

        for dataset in d_manager.config_yml["datasets"]:
            for datum_dataset in dataset["datum_datasets"]:
                len(datum_dataset_result)
                utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    def test_select_bbox(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        bboxes:
          - bbox: stickman

        exec: selection_module.any_of_annotations(item, select_yml)
"""
        )

        def test_categories():
            return {AnnotationType.label: LabelCategories.from_iterable(["stickman", "animal"])}

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=1, label=1)],
                ),
            ],
            categories=test_categories(),
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 1)
        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Bbox(0, 1, 2, 3, id=0, label=0)],
                )
            ],
            categories=test_categories(),
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    def test_select_label_with_attr(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: stickman
            attr:
              animation_quality: good

        exec: selection_module.any_of_annotations(item, select_yml)
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
                    annotations=[dm.Label(id=1, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000002",
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
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 1)
        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000002",
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

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    def test_select_label_with_attr_not_support(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: stickman
            attr:
              animation_quality: good

        exec: not selection_module.any_of_annotations(item, select_yml)
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
                    annotations=[dm.Label(id=1, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000002",
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
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)
        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000001",
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

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    def test_select_label_with_multi_attrs(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
    - select:
      labels:
      - label: stickman
        attr:
          animation_quality: good
          occluded: True

      exec: selection_module.any_of_annotations(item, select_yml)
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
                    annotations=[dm.Label(id=1, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1, attributes={"occluded": False})],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=3, label=1, attributes={"occluded": True})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Label(id=4, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Label(id=5, label=1, attributes={"animation_quality": "good", "occluded": True, "id": 1})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Label(id=6, label=1, attributes={"animation_quality": "good", "occluded": False})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Label(id=7, label=1, attributes={"animation_quality": "good", "occluded": True})],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[dm.Label(id=8, label=1, attributes={"animation_quality": "bad", "occluded": True})],
                ),
                dm.DatasetItem(
                    id="frame_000009",
                    annotations=[dm.Label(id=9, label=0, attributes={"animation_quality": "good", "occluded": True})],
                ),
                dm.DatasetItem(
                    id="frame_000010",
                    annotations=[
                        dm.Label(
                            id=10,
                            label=1,
                            attributes={
                                "animation_quality": "good",
                                "occluded": True,
                                "id": 1,
                                "extra": "anno info",
                            },
                        )
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman"],
                    ["animation_quality", "occluded", "id", "extra"],
                )
            },
        )
        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 11)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)
        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Label(id=5, label=1, attributes={"animation_quality": "good", "occluded": True, "id": 1})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Label(id=7, label=1, attributes={"animation_quality": "good", "occluded": True})],
                ),
                dm.DatasetItem(
                    id="frame_000010",
                    annotations=[
                        dm.Label(
                            id=10,
                            label=1,
                            attributes={
                                "animation_quality": "good",
                                "occluded": True,
                                "id": 1,
                                "extra": "anno info",
                            },
                        )
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman"],
                    ["animation_quality", "occluded", "id", "extra"],
                )
            },
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    # TODO: add log how many label are selected or how many deleted
    def test_select_label_attr_array_support(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: stickman
            attr:
              animation_quality: [good, normal]

        exec: selection_module.any_of_annotations(item, select_yml)
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
                    annotations=[dm.Label(id=4, label=1, attributes={"animation_quality": "normal"})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Label(id=5, label=1, attributes={"animation_quality": "bad"})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Label(id=6, label=1, attributes={"occluded": True})],
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
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 7)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)
        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=3, label=1, attributes={"animation_quality": "good"})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Label(id=4, label=1, attributes={"animation_quality": "normal"})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman"],
                    ["animation_quality", "occluded"],
                )
            },
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    def test_select_and_mechanism(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: key_point_animated
        exec: selection_module.any_of_annotations(item, select_yml)
      - select:
        polygons:
          - polygon: stickman_stationary
        exec: selection_module.any_of_annotations(item, select_yml)
"""
        )

        def test_categories():
            return {AnnotationType.label: LabelCategories.from_iterable(["key_point_animated", "stickman_stationary", "key_point_animated_1"])}

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(id="frame_000000", annotations=[dm.Label(id=1, label=0)]),
                dm.DatasetItem(id="frame_000001", annotations=[dm.Label(id=2, label=1)]),
                dm.DatasetItem(
                    id="frame_000003", annotations=[dm.Polygon(id=3, points=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], label=1), dm.Label(id=4, label=0)]
                ),
                dm.DatasetItem(id="frame_000004", annotations=[dm.Label(id=4, label=2)]),
                dm.DatasetItem(id="frame_000005", annotations=[dm.Label(id=5, label=1)]),
                dm.DatasetItem(
                    id="frame_000006", annotations=[dm.Polygon(id=6, points=[1.0, 5.0, 7.0, 4.0, 5.0, 6.0], label=1), dm.Label(id=7, label=2)]
                ),
            ],
            categories=test_categories(),
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)
        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 6)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 1)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000003", annotations=[dm.Polygon(id=3, points=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], label=1), dm.Label(id=4, label=0)]
                )
            ],
            categories=test_categories(),
        )
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    def test_select_and_mechanism_with_attr(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: key_point_animated
        exec: selection_module.any_of_annotations(item, select_yml)
      - select:
        labels:
          - label: stickman
            attr:
              animation_quality : [good, normal]
              occluded: False
        exec: selection_module.any_of_annotations(item, select_yml)
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
                    annotations=[dm.Label(id=1, label=1)],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1, attributes={"occluded": False}), dm.Label(id=3, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=4, label=1, attributes={"animation_quality": "good", "occluded": False}), dm.Label(id=5, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Label(id=6, label=1, attributes={"animation_quality": "normal", "occluded": False}), dm.Label(id=7, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Label(id=8, label=1, attributes={"animation_quality": "good", "occluded": True, "id": 1})],
                ),
                dm.DatasetItem(
                    id="frame_000006",
                    annotations=[dm.Label(id=9, label=1, attributes={"animation_quality": "good", "occluded": False})],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Label(id=10, label=1, attributes={"animation_quality": "good", "occluded": False}), dm.Label(id=11, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000008",
                    annotations=[dm.Label(id=12, label=1, attributes={"animation_quality": "bad", "occluded": True})],
                ),
                dm.DatasetItem(
                    id="frame_000009",
                    annotations=[dm.Label(id=13, label=1, attributes={"animation_quality": "good", "occluded": True})],
                ),
                dm.DatasetItem(
                    id="frame_000010",
                    annotations=[
                        dm.Label(
                            id=10,
                            label=1,
                            attributes={
                                "animation_quality": "good",
                                "occluded": True,
                                "id": 1,
                                "extra": "anno info",
                            },
                        )
                    ],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman"],
                    ["animation_quality", "occluded", "id", "extra"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)

        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 11)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 3)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=4, label=1, attributes={"animation_quality": "good", "occluded": False}), dm.Label(id=5, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Label(id=6, label=1, attributes={"animation_quality": "normal", "occluded": False}), dm.Label(id=7, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000007",
                    annotations=[dm.Label(id=10, label=1, attributes={"animation_quality": "good", "occluded": False}), dm.Label(id=11, label=0)],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman"],
                    ["animation_quality", "occluded", "id", "extra"],
                )
            },
        )
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    def test_select_or_mechanism(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: stickman_stationary
            attr:
              visible_body_shape: True
              semi_visible : False
          - label: stickman_stationary
            attr:
              visible_body_shape: False
              semi_visible : True
        exec: selection_module.any_of_annotations(item, select_yml)
"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=1, label=1, attributes={"visible_body_shape": True, "semi_visible": False})],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1, attributes={"visible_body_shape": False, "semi_visible": False})],
                ),
                dm.DatasetItem(
                    id="frame_000003",
                    annotations=[dm.Label(id=3, label=1, attributes={"visible_body_shape": True, "semi_visible": True})],  # wrong annotation
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Label(id=4, label=1, attributes={"visible_body_shape": False, "semi_visible": True})],
                ),
                dm.DatasetItem(
                    id="frame_000005",
                    annotations=[dm.Label(id=5, label=0)],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman_stationary"],
                    ["visible_body_shape", "semi_visible"],
                )
            },
        )

        test_config["datasets"][0]["datum_datasets"] = [datum_dataset_org]
        d_manager = DatasetManager(config_yml=test_config)

        self.assertEqual(len(d_manager.config_yml["datasets"]), 1)
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 5)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        datum_dataset_result = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000001",
                    annotations=[dm.Label(id=1, label=1, attributes={"visible_body_shape": True, "semi_visible": False})],
                ),
                dm.DatasetItem(
                    id="frame_000004",
                    annotations=[dm.Label(id=4, label=1, attributes={"visible_body_shape": False, "semi_visible": True})],
                ),
            ],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman_stationary"],
                    ["visible_body_shape", "semi_visible"],
                )
            },
        )
        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)

    def test_select_no_match(self):
        test_config = yaml.safe_load(
            """
datasets:
  - dataset:
    selects:
      - select:
        labels:
          - label: stickman
            attr:
              animation_quality: good

        exec: selection_module.any_of_annotations(item, select_yml)
      - select:
        polygons:
          - polygon: stickman
            attr:
              animation_quality: [visible_body_shape, semi_visible]
        exec: selection_module.any_of_annotations(item, select_yml)

    generator:
      pollute_back: 4
      pollute_step: 4

"""
        )

        datum_dataset_org = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id="frame_000000",
                    annotations=[dm.Label(id=0, label=0)],
                ),
                dm.DatasetItem(
                    id="frame_000002",
                    annotations=[dm.Label(id=2, label=1, attributes={"animation_quality": "bad"})],
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
        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 2)

        d_manager.process()

        self.assertEqual(len(d_manager.config_yml["datasets"][0]["datum_datasets"][0]), 0)
        datum_dataset_result = dm.Dataset.from_iterable(
            [],
            categories={
                AnnotationType.label: utils.generate_label_categories(
                    ["key_point_animated", "stickman"],
                    ["animation_quality"],
                )
            },
        )

        for datum_dataset in d_manager.config_yml["datasets"][0]["datum_datasets"]:
            utils.compare_datumaro_datasets(self, datum_dataset, datum_dataset_result)
