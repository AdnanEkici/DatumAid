from __future__ import annotations

import copy
import json
import os
import shutil
import unittest
import xml.etree.ElementTree as ET
from uuid import uuid4

import cv2
import requests
from parameterized import parameterized
from requests.compat import urljoin

import datumaid.utils as dataset_utils
from datumaid.api_handler import CVATHandler
from test import utils as test_utils


class TestCvatApi(unittest.TestCase):
    url = None
    token = None
    if "CVAT_URL" in os.environ and "CVAT_USER_TOKEN" in os.environ:
        url = os.environ["CVAT_URL"]
        token = os.environ["CVAT_USER_TOKEN"]
    cvat_host_config = {
        "organization": "test",
        "url": url,
        "token": token,
    }

    old_project_id = None
    cvat_host = cvat_host_config["url"]
    cvat_api_token = cvat_host_config["token"]
    cvat_organization = cvat_host_config["organization"]

    api_handler = CVATHandler(host_url=cvat_host_config["url"], api_token=cvat_host_config["token"])
    test_path = os.sep + "tmp" + os.sep + f"api_test_{uuid4()}"
    export_path = os.path.join(test_path, "export")
    original_path = os.path.join(test_path, "original")
    auth_headers = {"Authorization": f"Token {cvat_api_token}"}
    labels = [
        {"name": "key_point_animated", "attributes": []},
        {"name": "stickman", "attributes": [{"name": "animation_quality", "input_type": "text", "mutable": False, "values": []}]},
    ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_path)
        cls.api_handler.delete_project(id=cls.old_project_id)

    def setUp(self):
        self.api_handler = CVATHandler(host_url=self.cvat_host_config["url"], api_token=self.cvat_host_config["token"])

    @classmethod
    def setUpClass(cls):
        if cls.url is None or cls.token is None:
            raise unittest.SkipTest(
                "Skipping tests because gitlab runner cvat token or url is None. Either add url "
                "and token to your environment variables or run tests from pipeline. For gitlab "
                "runner url and token please ask project maintainers."
            )

        os.makedirs(cls.export_path, exist_ok=True)
        os.makedirs(cls.original_path, exist_ok=True)

        cls.old_project_id = cls.api_handler.create_project(name="Old Project Name", labels=cls.labels, organization=cls.cvat_organization)

    def check_project_id(self, task_id, project_id):
        r = requests.get(urljoin(self.cvat_host, f"api/tasks/{task_id}"), headers=self.auth_headers)
        task_project_id = json.loads(r.text)["project_id"]
        self.assertEqual(task_project_id, project_id)

    def check_if_exists(self, id=None, name="", exists=True, type="projects"):
        if exists:
            r = requests.get(urljoin(self.cvat_host, f"api/{type}/{id}"), headers=self.auth_headers)
            self.assertEqual(r.status_code, 200)
        else:
            r = requests.get(urljoin(self.cvat_host, f"api/{type}"), headers=self.auth_headers)
            results = json.loads(r.text)["results"]
            self.assertEqual(len([i for i in results if i["name"] == name]), 0)

        return json.loads(r.text)

    def test_project_creation(self):
        project_name = "test_project_creation"
        self.assertRaises(CVATHandler.HttpError, self.api_handler.create_project, project_name, None, None)
        self.check_if_exists(name=project_name, exists=False, type="projects")
        self.assertRaises(CVATHandler.HttpError, self.api_handler.create_project, project_name, self.labels, None)
        self.check_if_exists(name=project_name, exists=False, type="projects")
        self.assertRaises(CVATHandler.HttpError, self.api_handler.create_project, None, None, None)
        self.check_if_exists(exists=False, type="projects")
        prj_id = self.api_handler.create_project(name=project_name, labels=self.labels, organization=self.cvat_organization)
        self.check_if_exists(id=prj_id, exists=True, type="projects")
        self.api_handler.delete_project(prj_id)
        self.check_if_exists(name=project_name, exists=False, type="projects")
        self.assertRaises(CVATHandler.HttpError, self.api_handler.delete_project, prj_id)

    def test_task_creation(self):
        project_name = "test_task_creation_project"
        task_name = "test_task_creation"

        # organization is not provided
        self.assertRaises(Exception, self.api_handler.create_task, task_name, None, "", 1, [], 200)
        self.check_if_exists(name=task_name, exists=False, type="tasks")

        # project id is not provided
        self.assertRaises(Exception, self.api_handler.create_task, task_name, "test", "", None, [], 200)
        self.check_if_exists(name=task_name, exists=False, type="tasks")

        prj_id = self.api_handler.create_project(name=project_name, labels=self.labels, organization=self.cvat_organization)
        self.check_if_exists(id=prj_id, exists=True, type="projects")
        task_id = self.api_handler.create_task(task_name)  # uses initialized project id and organization
        self.check_if_exists(id=task_id, exists=True, type="tasks")
        individual_task_id = self.api_handler.create_task(task_name, organization="test", project_id=self.old_project_id)  # uses given values
        self.check_project_id(individual_task_id, self.old_project_id)  # Check if task created in given project

        self.api_handler.delete_task(task_id)
        self.api_handler.delete_task(individual_task_id)
        self.check_if_exists(name=project_name, exists=False, type="tasks")
        self.api_handler.delete_project(prj_id)
        self.check_if_exists(name=project_name, exists=False, type="projects")

    def step_create_task_from_dataset(self, is_video=False):
        task_name = "Test task created via API"
        proj_name = "Test project created via API"
        api_handler = CVATHandler(host_url=self.cvat_host, api_token=self.cvat_api_token)
        proj_id = api_handler.create_project(name=proj_name, labels=self.labels, organization=self.cvat_organization)
        test_utils.create_datumaro_dataset(path=self.original_path, labels=self.labels)
        task_id = api_handler.create_task(task_name)
        upload_file_path = os.path.join(self.original_path, "test_video.mp4") if is_video else os.path.join(self.original_path, "images")
        api_handler.upload_files(task_id=task_id, file_path=upload_file_path, is_video=is_video, image_quality=100)
        api_handler.upload_annotations(task_id=task_id, file_path=os.path.join(self.original_path, "default.xml"))
        response_body = self.check_if_exists(id=task_id, exists=True, type="tasks")
        self.assertEqual(len(response_body["labels"]), len(self.labels))

        return task_id, proj_id, task_name, proj_name

    def step_test_compare_hashes(self, is_video=False):
        original_images = (
            dataset_utils.get_images(os.path.join(self.original_path, "video_frames"))
            if is_video
            else dataset_utils.get_images(os.path.join(self.original_path, "images"))
        )
        original_annotations = dataset_utils.get_targetted_files(self.original_path, target_extension_list=["xml"])[0]
        exported_images = dataset_utils.get_images(os.path.join(self.export_path, "images"))
        exported_annotations = dataset_utils.get_targetted_files(self.export_path, target_extension_list=["xml"])[0]

        if len(original_images) > 0 and len(exported_images) > 0:
            [
                test_utils.compare_images(self, cv2.imread(original), cv2.imread(exported), 5)
                for original, exported in zip(original_images, exported_images)
            ]

        self.compare_annotations(original_annotations, exported_annotations)

    def compare_annotations(self, original_file, exported_file):
        original_images = ET.parse(original_file).getroot().findall("image")
        exported_images = ET.parse(exported_file).getroot().findall("image")

        def compare(o, e, tag, attribute):
            self.assertEqual(o.find(tag).get(attribute), e.find(tag).get(attribute))

        for original, exported in zip(original_images, exported_images):
            self.assertEqual(original.get("id"), exported.get("id"))
            self.assertEqual(original.get("width"), exported.get("width"))
            self.assertEqual(original.get("height"), exported.get("height"))

            compare(original, exported, "tag", "label")
            compare(original, exported, "polygon", "label")
            compare(original, exported, "polygon", "points")
            compare(original, exported, "box", "label")
            compare(original, exported, "box", "xtl")
            compare(original, exported, "box", "ytl")
            compare(original, exported, "box", "xbr")
            compare(original, exported, "box", "ybr")

    def export_task(self, task_id, export_type="dataset", format="CVAT for images 1.1"):
        zipfile_name = self.api_handler.export_task(task_id=task_id, export_type=export_type, export_path=self.export_path, format=format)
        dataset_utils.extract_zipfile(export_path=self.export_path, filepath=os.path.join(self.export_path, zipfile_name))

    @parameterized.expand([("import_video", True), ("import_images", False)])
    def test_create_task_from_given_dataset(self, _, is_video):
        task_id, proj_id, task_name, proj_name = self.step_create_task_from_dataset(is_video)
        self.export_task(task_id)
        self.step_test_compare_hashes(is_video)
        self.api_handler.delete_task(task_id=task_id)
        self.api_handler.delete_project(id=proj_id)

    def test_change_existing_task_labels(self):
        task_id, proj_id, task_name, proj_name = self.step_create_task_from_dataset()
        self.export_task(task_id, export_type="annotations")  # will overwrite previous annotation
        annotations_file_path = dataset_utils.get_targetted_files(self.export_path, target_extension_list=["xml"])[0]
        annotations = ET.parse(annotations_file_path)
        old_annoations = copy.deepcopy(annotations)
        # change an attribute from first image
        attr_to_change = annotations.getroot().findall("image")[0].find("polygon")
        attr_to_change.set("occluded", "changed")

        # save old and new annotations
        new_annotations_path = os.sep + "tmp" + os.sep + "new_annotations_test.xml"
        old_annotations_path = os.sep + "tmp" + os.sep + "old_annotations_test.xml"
        annotations.write(new_annotations_path)
        old_annoations.write(old_annotations_path)

        self.api_handler.upload_annotations(task_id=task_id, file_path=new_annotations_path)
        self.export_task(task_id, export_type="annotations")  # will overwrite previous annotation

        # exporting will overwrite old file but the path will remain the same
        self.assertRaises(Exception, self.compare_annotations(annotations_file_path, old_annotations_path))

        os.remove(new_annotations_path)
        os.remove(old_annotations_path)

        self.api_handler.delete_task(task_id=task_id)
        self.api_handler.delete_project(id=proj_id)

    def test_project_export(self):
        self.api_handler.export_project(
            project_id=1,
            export_type="annotations",
            export_path=self.test_path + "_download",
            format="CVAT for images 1.1",
            exclude_list=[95, 94, 93, 92],
        )
        expected_annotations = ["annotations_96.zip", "annotations_97.zip"]
        downloaded_annotations = [
            os.path.basename(i) for i in dataset_utils.get_targetted_files(self.test_path + "_download", target_extension_list=[".zip"])
        ]

        self.assertListEqual(
            expected_annotations,
            downloaded_annotations,
            msg=f"Assertion failed. Expected List: {expected_annotations} is not same with downloaded list {downloaded_annotations}",
        )
