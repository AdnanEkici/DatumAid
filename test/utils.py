from __future__ import annotations

import copy
import hashlib
import os
import random
import threading
import time
import tracemalloc
from enum import Enum

import cv2
import datumaro as dm
import matplotlib.pyplot as plt
import numpy as np
import psutil
from datumaro.components.annotation import AnnotationType
from datumaro.components.annotation import LabelCategories
from datumaro.util.test_utils import compare_datasets

import datumaid.utils as dataset_utils


def create_sample_video(video_path, size=(1080, 1920), fps=25):
    video_name = os.path.join(video_path, "pollute.mp4")

    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[1], size[0]), 0)
    for idx in range(25, 1000, 25):
        data = np.zeros(size, dtype="uint8")
        data = cv2.circle(
            data,
            (idx, idx),
            10,
            255,
            -1,
        )
        out.write(data)

    out.release()

    return video_name


def compare_sample_video_frame_ids(image, id):
    frame_idx = id if type(id) == int else int(id.split("_")[1])

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255
    image = np.asarray(image, dtype=np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = cv2.approxPolyDP(contours[0], 3, True)
    centers, _ = cv2.minEnclosingCircle(contours_poly)
    # In created video circles moves 25 pixel in every frame
    # We gived 8 pixel error due to the video compression
    img_idx = int((centers[0] + 8) // 25) - 1
    return img_idx == frame_idx


def generate_test_image(width=1920, height=1080, channel=3):
    def gray(width, height):
        gray_img = np.zeros((height, width), dtype=np.uint8)
        val = 0
        bar_width = 120
        step = 17
        for i in range(0, width, bar_width):
            gray_img[0:height, i : i + bar_width] = val  # noqa
            val += step
        return gray_img

    def color(width, height):
        class TestColors(Enum):
            WHITE = (255, 255, 255)
            YELLOW = (0, 255, 255)
            CYAN = (255, 255, 0)
            GREEN = (0, 255, 0)
            MAGENTA = (255, 0, 255)
            RED = (0, 0, 255)
            BLUE = (255, 0, 0)
            BLACK = (0, 0, 0)

        bar_width = 240
        color_img = np.zeros((height, width, 3), dtype=np.uint8)
        i = 0
        for c in TestColors:
            color_img[0:height, i : i + bar_width] = c.value  # noqa
            i += bar_width
        return color_img

    if channel == 3:
        return color(width, height)
    return gray(width, height)


def compare_datumaro_datasets(self, datum_dataset_1, datum_dataset_2):
    compare_datasets(self, datum_dataset_1, copy.deepcopy(datum_dataset_2))


def compare_images(self, image1, image2, confidence=40):
    self.assertEqual(image1.shape, image2.shape)
    difference1 = cv2.subtract(image1, image2)
    ret, difference1 = cv2.threshold(difference1, confidence, 0, cv2.THRESH_BINARY)
    self.assertFalse(np.any(difference1))
    difference2 = cv2.subtract(image2, image1)
    ret, difference2 = cv2.threshold(difference2, confidence, 0, cv2.THRESH_TOZERO)
    self.assertFalse(np.any(difference2))


def generate_label_categories(labels, attrs):
    label_categories = LabelCategories(attributes=attrs)
    for label in labels:
        label_categories.add(label)
    return label_categories


def generator_test_aa1(tmp_dir):
    return dm.Dataset.from_iterable(
        [
            dm.DatasetItem(
                image=dm.Image(path=tmp_dir + "test/data/datumaro-aa1/images/default/frame_000000.PNG"),
                id="frame_000000",
                attributes={"frame": 0},
                annotations=[
                    dm.Label(id=0, label=0, attributes={"is_running": 0.0}, group=0),
                    dm.Bbox(
                        578.33,
                        550.31,
                        42.40999999999997,
                        59.120000000000005,
                        label=1,
                        id=0,
                        group=0,
                        attributes={
                            "custom": "",
                            "animation_quality": "normal",
                            "colored": False,
                            "occluded": False,
                            "rotation": 0.0,
                            "track_id": 0,
                            "keyframe": True,
                        },
                    ),
                    dm.Polygon(
                        [
                            625.37,
                            1080.0,
                            686.28,
                            1078.51,
                            1237.5,
                            1005.3,
                            1492.3,
                            861.5,
                            1631.3,
                            850.6,
                            1716.96,
                            956.42,
                            1751.01,
                            1080.0,
                            1875.12,
                            1080.0,
                            1919.63,
                            1079.15,
                            1919.24,
                            719.82,
                            1905.88,
                            663.02,
                            1869.12,
                            577.3,
                            1749.09,
                            421.93,
                            1644.74,
                            222.09,
                            1517.09,
                            0.0,
                            786.73,
                            0.0,
                            3.61,
                            2.33,
                            5.13,
                            1080.0,
                        ],
                        id=0,
                        attributes={"occluded": False, "track_id": 1, "keyframe": True},
                        group=0,
                        label=2,
                        z_order=0,
                    ),
                    dm.Polygon(
                        [
                            718.4,
                            270.2,
                            729.98,
                            317.7,
                            746.68,
                            299.71,
                            1177.2,
                            294.57,
                            1193.91,
                            312.56,
                            1200.34,
                            279.15,
                            1164.35,
                            266.3,
                            958.73,
                            258.59,
                        ],
                        id=0,
                        attributes={"occluded": False, "track_id": 2, "keyframe": True},
                        group=0,
                        label=3,
                        z_order=0,
                    ),
                ],
            ),
            dm.DatasetItem(
                image=dm.Image(path=tmp_dir + "test/data/datumaro-aa1/images/default/frame_000001.PNG"),
                id="frame_000001",
                attributes={"frame": 1},
                annotations=[
                    dm.Label(id=0, label=0, attributes={"is_running": 0.0}, group=0),
                    dm.Bbox(
                        634.88,
                        549.03,
                        42.40999999999997,
                        59.110000000000014,
                        label=1,
                        id=0,
                        group=0,
                        attributes={
                            "custom": "",
                            "animation_quality": "normal",
                            "colored": False,
                            "occluded": False,
                            "rotation": 0.0,
                            "track_id": 0,
                            "keyframe": True,
                        },
                    ),
                    dm.Polygon(
                        [
                            625.37,
                            1080.0,
                            686.28,
                            1078.51,
                            1237.5,
                            1005.3,
                            1492.3,
                            861.5,
                            1631.3,
                            850.6,
                            1716.96,
                            956.42,
                            1751.01,
                            1080.0,
                            1875.12,
                            1080.0,
                            1919.63,
                            1079.15,
                            1919.24,
                            719.82,
                            1905.88,
                            663.02,
                            1869.12,
                            577.3,
                            1749.09,
                            421.93,
                            1644.74,
                            222.09,
                            1517.09,
                            0.0,
                            786.73,
                            0.0,
                            3.61,
                            2.33,
                            5.13,
                            1080.0,
                        ],
                        id=0,
                        attributes={
                            "occluded": False,
                            "track_id": 1,
                            "keyframe": False,
                        },
                        group=0,
                        label=2,
                        z_order=0,
                    ),
                    dm.Polygon(
                        [
                            718.4,
                            270.2,
                            729.98,
                            317.7,
                            746.68,
                            299.71,
                            1177.2,
                            294.57,
                            1193.91,
                            312.56,
                            1200.34,
                            279.15,
                            1164.35,
                            266.3,
                            958.73,
                            258.59,
                        ],
                        id=0,
                        attributes={
                            "occluded": False,
                            "track_id": 2,
                            "keyframe": False,
                        },
                        group=0,
                        label=3,
                        z_order=0,
                    ),
                ],
            ),
            dm.DatasetItem(
                image=dm.Image(path=tmp_dir + "test/data/datumaro-aa1/images/default/frame_000002.PNG"),
                id="frame_000002",
                attributes={"frame": 2},
                annotations=[
                    dm.Label(id=0, label=0, attributes={"is_running": 0.0}, group=0),
                    dm.Bbox(
                        704.27,
                        546.46,
                        42.40999999999997,
                        59.110000000000014,
                        label=1,
                        id=0,
                        group=0,
                        attributes={
                            "custom": "",
                            "animation_quality": "normal",
                            "colored": False,
                            "occluded": False,
                            "rotation": 0.0,
                            "track_id": 0,
                            "keyframe": True,
                        },
                    ),
                    dm.Polygon(
                        [
                            625.37,
                            1080.0,
                            686.28,
                            1078.51,
                            1237.5,
                            1005.3,
                            1492.3,
                            861.5,
                            1631.3,
                            850.6,
                            1716.96,
                            956.42,
                            1751.01,
                            1080.0,
                            1875.12,
                            1080.0,
                            1919.63,
                            1079.15,
                            1919.24,
                            719.82,
                            1905.88,
                            663.02,
                            1869.12,
                            577.3,
                            1749.09,
                            421.93,
                            1644.74,
                            222.09,
                            1517.09,
                            0.0,
                            786.73,
                            0.0,
                            3.61,
                            2.33,
                            5.13,
                            1080.0,
                        ],
                        id=0,
                        attributes={
                            "occluded": False,
                            "track_id": 1,
                            "keyframe": False,
                        },
                        group=0,
                        label=2,
                        z_order=0,
                    ),
                    dm.Polygon(
                        [
                            718.4,
                            270.2,
                            729.98,
                            317.7,
                            746.68,
                            299.71,
                            1177.2,
                            294.57,
                            1193.91,
                            312.56,
                            1200.34,
                            279.15,
                            1164.35,
                            266.3,
                            958.73,
                            258.59,
                        ],
                        id=0,
                        attributes={
                            "occluded": False,
                            "track_id": 2,
                            "keyframe": False,
                        },
                        group=0,
                        label=3,
                        z_order=0,
                    ),
                ],
            ),
            dm.DatasetItem(
                image=dm.Image(path=tmp_dir + "test/data/datumaro-aa1/images/default/frame_000003.PNG"),
                id="frame_000003",
                attributes={"frame": 3},
                annotations=[
                    dm.Label(id=0, label=0, attributes={"is_running": 0.0}, group=0),
                    dm.Bbox(
                        760.82,
                        542.6,
                        42.40999999999997,
                        59.120000000000005,
                        label=1,
                        id=0,
                        group=0,
                        attributes={
                            "custom": "",
                            "animation_quality": "normal",
                            "colored": False,
                            "occluded": False,
                            "rotation": 0.0,
                            "track_id": 0,
                            "keyframe": True,
                        },
                    ),
                    dm.Polygon(
                        [
                            625.37,
                            1080.0,
                            686.28,
                            1078.51,
                            1237.5,
                            1005.3,
                            1492.3,
                            861.5,
                            1631.3,
                            850.6,
                            1716.96,
                            956.42,
                            1751.01,
                            1080.0,
                            1875.12,
                            1080.0,
                            1919.63,
                            1079.15,
                            1919.24,
                            719.82,
                            1905.88,
                            663.02,
                            1869.12,
                            577.3,
                            1749.09,
                            421.93,
                            1644.74,
                            222.09,
                            1517.09,
                            0.0,
                            786.73,
                            0.0,
                            3.61,
                            2.33,
                            5.13,
                            1080.0,
                        ],
                        id=0,
                        attributes={
                            "occluded": False,
                            "track_id": 1,
                            "keyframe": False,
                        },
                        group=0,
                        label=2,
                        z_order=0,
                    ),
                    dm.Polygon(
                        [
                            718.4,
                            270.2,
                            729.98,
                            317.7,
                            746.68,
                            299.71,
                            1177.2,
                            294.57,
                            1193.91,
                            312.56,
                            1200.34,
                            279.15,
                            1164.35,
                            266.3,
                            958.73,
                            258.59,
                        ],
                        id=0,
                        attributes={
                            "occluded": False,
                            "track_id": 2,
                            "keyframe": False,
                        },
                        group=0,
                        label=3,
                        z_order=0,
                    ),
                ],
            ),
            dm.DatasetItem(
                image=dm.Image(path=tmp_dir + "test/data/datumaro-aa1/images/default/frame_000004.PNG"),
                id="frame_000004",
                attributes={"frame": 4},
                annotations=[
                    dm.Label(id=0, label=0, attributes={"is_running": 0.0}, group=0),
                    dm.Bbox(
                        843.07,
                        542.6,
                        42.40999999999997,
                        59.120000000000005,
                        label=1,
                        id=0,
                        group=0,
                        attributes={
                            "custom": "",
                            "animation_quality": "normal",
                            "colored": False,
                            "occluded": False,
                            "rotation": 0.0,
                            "track_id": 0,
                            "keyframe": True,
                        },
                    ),
                    dm.Polygon(
                        [
                            625.37,
                            1080.0,
                            686.28,
                            1078.51,
                            1237.5,
                            1005.3,
                            1492.3,
                            861.5,
                            1631.3,
                            850.6,
                            1716.96,
                            956.42,
                            1751.01,
                            1080.0,
                            1875.12,
                            1080.0,
                            1919.63,
                            1079.15,
                            1919.24,
                            719.82,
                            1905.88,
                            663.02,
                            1869.12,
                            577.3,
                            1749.09,
                            421.93,
                            1644.74,
                            222.09,
                            1517.09,
                            0.0,
                            786.73,
                            0.0,
                            3.61,
                            2.33,
                            5.13,
                            1080.0,
                        ],
                        id=0,
                        attributes={
                            "occluded": False,
                            "track_id": 1,
                            "keyframe": False,
                        },
                        group=0,
                        label=2,
                        z_order=0,
                    ),
                    dm.Polygon(
                        [
                            718.4,
                            270.2,
                            729.98,
                            317.7,
                            746.68,
                            299.71,
                            1177.2,
                            294.57,
                            1193.91,
                            312.56,
                            1200.34,
                            279.15,
                            1164.35,
                            266.3,
                            958.73,
                            258.59,
                        ],
                        id=0,
                        attributes={
                            "occluded": False,
                            "track_id": 2,
                            "keyframe": False,
                        },
                        group=0,
                        label=3,
                        z_order=0,
                    ),
                ],
            ),
        ],
        categories={
            AnnotationType.label: generate_label_categories(
                ["key_point_animated", "stickman", "scene_forest", "mask"],
                ["custom", "is_running", "occluded", "animation_quality", "colored"],
            )
        },
    )


def show_and_save_image(function):
    def inner1(image_path, *args, **kwargs):
        original_image, altered_image = function(image_path, *args, **kwargs)
        if kwargs["show_image"]:
            cv2.imshow("Original", original_image)
            cv2.imshow("Altered", altered_image)
            cv2.waitKey(0)
        if kwargs["overwrite_image"]:
            try:
                altered_image = cv2.cvtColor(altered_image, cv2.COLOR_BGR2RGB)
            except cv2.error:
                altered_image = altered_image
            plt.imsave(image_path, altered_image)

    return inner1


@show_and_save_image
def add_noise_to_image(
    image_path: str,
    noise_type: str,
    mean_for_gaussian=0,
    standard_deviation_for_gaussian_noise=0.1,
    scale_parameter_for_rayleigh_noise=0.1,
    overwrite_image=False,
    show_image=False,
):
    image = cv2.imread(image_path)

    def add_rayleigh_noise():
        row, col, ch = image.shape
        s = np.random.rayleigh(scale_parameter_for_rayleigh_noise, (row, col, ch))
        noisy = image + s
        noisy = np.clip(noisy, 0, 1)

        return image, noisy

    def add_gaussian_noise():
        row, col, ch = image.shape
        mean = mean_for_gaussian
        var = standard_deviation_for_gaussian_noise
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)  # noqa
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 1)
        return image, noisy

    def add_salt_and_pepper_noise():
        noisy = image
        row, col, _ = noisy.shape

        number_of_pixels = random.randint(50, 100)
        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)

            x_coord = random.randint(0, col - 1)

            # Color that pixel to white
            noisy[y_coord][x_coord] = 255

        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)

            x_coord = random.randint(0, col - 1)

            noisy[y_coord][x_coord] = 0

        return image, noisy

    def add_poisson_noise():
        img = image

        img_noisy = np.random.poisson(img / 255.0 * 10) * 10
        img_noisy = np.clip(img_noisy, 0, 255)
        img_noisy = img_noisy.astype(np.uint8)

        return img, img_noisy

    selector = {
        "s&p": add_salt_and_pepper_noise(),
        "gaussian": add_gaussian_noise(),
        "poisson": add_poisson_noise(),
        "rayleigh": add_rayleigh_noise(),
    }
    return selector[noise_type]


@show_and_save_image
def rotate_image(image_path: str, rotation_degree: float, show_image=False, overwrite_image=False, interpolation=None):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D((cX, cY), rotation_degree, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return image, rotated


@show_and_save_image
def resize_image(
    image_path: str, resize_ratio: int, show_image=False, overwrite_image=False, return_to_original_size=False, interpolation=cv2.INTER_NEAREST
):
    original_image = cv2.imread(image_path)
    resized_image = original_image
    # Get the original shape of the image
    height, width = resized_image.shape[:2]

    # Resize the image by a certain percentage
    percentage = resize_ratio
    resized_image = cv2.resize(resized_image, (int(width * percentage), int(height * percentage)), interpolation=interpolation)

    if return_to_original_size:
        resized_image = cv2.resize(resized_image, (width, height), interpolation=interpolation)

    return original_image, resized_image


@show_and_save_image
def adjust_contrast(image_path: str, show_image=False, alpha=1.5, beta=10, overwrite_image=False):
    image = cv2.imread(image_path)

    alpha = alpha  # Contrast control (1.0-3.0)
    beta = beta  # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return image, adjusted


def generator_test_aa1_small():
    return dm.Dataset.from_iterable(
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
        categories=get_test_categories(),
    )


def get_test_categories():
    return {
        AnnotationType.label: generate_label_categories(
            ["camera_ir_wh", "camera_ir_bh"],
            ["animation_quality", "occluded"],
        )
    }


def sort_coordinates(points):
    """Sorts polygon points for cvat annotation

    Parameters
    ----------
    points : numpy array
        Polygon points of the dummy object

    Returns
    -------
    list
        Ordered points
    """
    current_index = 0
    ord_points = []
    while len(points) // 2 > 1:  # points array has 2 columns
        current_point = points[current_index]
        points = np.delete(points, current_index, 0)
        ord_points.append(current_point[0])
        ord_points.append(current_point[1])
        current_index, _ = min(enumerate(points), key=lambda x: np.linalg.norm(x[1] - current_point))

    return ord_points


def create_video_frames(fps, num_frames, path, size=(1920, 1080), fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"), video_name="test_video.mp4"):
    """Generates frames with dummy object in them then
    calculates polygons and bbox points of these objects. Finally,
    creates and saves a video that includes the generated frames.

    Returns
    -------
    tuple
        images: list of generated images
        polygon_list: list of sorted polygons of objects
        bbox_list: list of bboxes of objects
    """
    width, height = size
    x, y = 500, 500  # Dummy object start point
    radius = 50  # Dummy object radius
    step = 100  # Dummy jump size

    images = []
    polygon_list = []
    bbox_list = []
    video_path = os.path.join(path, video_name)
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for _ in range(num_frames):
        img = np.zeros((height, width), dtype=np.uint8)
        dummy_image = img.copy()

        cv2.circle(img, (x, y), radius, 255, thickness=-1)
        video.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

        cv2.circle(dummy_image, (x, y), radius, 255)
        polygon_points = np.transpose(np.where(dummy_image == 255))
        polygon = sort_coordinates(polygon_points)
        bbox = cv2.boundingRect(polygon_points)

        images.append(img)
        polygon_list.append(polygon)
        bbox_list.append(bbox)

        x += step
        y += step
    video.release()

    video_frames_path = os.path.join(path, "video_frames")
    os.makedirs(video_frames_path, exist_ok=True)
    export_video_frames(video_path=video_path, export_path=video_frames_path)

    return images, polygon_list, bbox_list


def compare_hashes(original_path, exported_path):
    md5hash_1 = hashlib.md5()
    md5hash_1.update(open(original_path, "rb").read())
    md5hash_2 = hashlib.md5()
    md5hash_2.update(open(exported_path, "rb").read())
    return md5hash_1.hexdigest() == md5hash_2.hexdigest()


def create_datumaro_dataset(path, labels):
    images, polygon_list, bbox_list = create_video_frames(fps=25, num_frames=3, path=path)
    dataset = dm.Dataset.from_iterable(
        [
            dm.DatasetItem(
                id=f"{i}",
                image=dm.Image(path=f"image{i}.png", data=images[i]),
                annotations=[
                    dm.Label(id=i, label=0),
                    dm.Polygon(polygon_list[i], id=i, label=1),
                    dm.Bbox(bbox_list[i][0], bbox_list[i][1], bbox_list[i][2], bbox_list[i][3], id=i, label=1),
                ],
            )
            for i in range(len(images))
        ],
        categories={
            AnnotationType.label: generate_label_categories(
                [label["name"] for label in labels],
                [attr["name"] for label in labels for attr in label["attributes"] if len(label["attributes"]) > 0],
            )
        },
    )

    dataset.export(save_dir=path, format="cvat", save_images=True)


def export_video_frames(video_path, export_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0

    while frame_index < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(export_path, f"frame_{frame_index}.png"), frame)
        frame_index += 1

    cap.release()


class PCFunctionBenchmarkUtilities:
    # TODO: Seperate this code.
    global usage_list
    global cpu_usage_before_calculation
    cpu_usage_before_calculation = 0  # noqa
    usage_list = []  # type: list[float] # noqa

    def profile_memory_usage(function):
        """Benchmarks the memory usage of given function as parameter.This decorator can get a message
        as parameter if given it will prompt it on terminal
        Parameters
        ----------
        function : function
            function that going to be benchmarked
        message : str, optional
            message for benchmark info, by default ""
        Returns
        -------
        float
            percentage of memory usage as megabytes
        """

        def inner1(*args, **kwargs):
            tracemalloc.start()
            function(*args, **kwargs)  # noqa
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 10**6
            message = kwargs.get("message", "")
            dataset_utils.get_logger().info(f"{message} peak memory usage was {peak_mb:.2f}MB")
            return peak_mb

        return inner1

    def profile_access_time(function):
        """Calculates access time to variable from function.This decorator can get a message
        as parameter if given it will prompt it on terminal

        Parameters
        ----------
        function : function
            function that going to be benchmarked
        Returns
        -------
        float
            seconds passed to get time
        """

        def inner1(*args, **kwargs):
            start_time = time.time()
            function(*args, **kwargs)  # noqa
            end_time = time.time()
            elapsed_time = end_time - start_time
            message = kwargs.get("message", "")
            dataset_utils.get_logger().info(message + " access time: " + str(elapsed_time) + " seconds\n")
            return elapsed_time

        return inner1

    def profile_cpu_usage(function):
        """Benchmarks the cpu usage of given function as parameter.This decorator can get a message
        as parameter if given it will prompt it on terminal

        Parameters
        ----------
        function : function
            function that going to be benchmarked
        message : str, optional
            message for benchmark info, by default ""
        Returns
        -------
        float
            percentage of cpu usage
            seconds to calculate percentage of cpu usage
        """

        def inner1(*args, **kwargs):
            message = kwargs.get("message", "")
            stop_event = threading.Event()
            PCFunctionBenchmarkUtilities().__start(stop_event)
            try:
                function(*args, **kwargs)  # noqa
            finally:
                try:
                    cpu_usage = abs(sum(usage_list) / len(usage_list))
                    occupied_iteration = len(usage_list)
                except ZeroDivisionError:
                    dataset_utils.get_logger().warning(message + " function is too fast to benchmark for cpu usage!")
                    cpu_usage = -1
                    occupied_iteration = -1

                if message and cpu_usage != -1:
                    cpu_usage_str = f"total cpu usage: {cpu_usage:.2f}%"
                    occupied_iteration_str = f"Cpu has been occupied {occupied_iteration} iteration"
                    message_str = f"{message} {cpu_usage_str}. {occupied_iteration_str}"
                    dataset_utils.get_logger().info(message_str)
            usage_list.clear()
            stop_event.set()
            return cpu_usage, occupied_iteration

        return inner1

    def __calculate_cpu_usage(self, cpu_usage_before_calculation, stop_event):
        cpu_iteration = 0
        while not stop_event.is_set():
            usage_list.append(psutil.cpu_percent(0.1) - cpu_usage_before_calculation)
            cpu_iteration = cpu_iteration + 1

    def __start(self, stop_event):
        cpu_usage_before_calculation = psutil.cpu_percent(1)
        time.sleep(1)
        cpu_thread = threading.Thread(target=self.__calculate_cpu_usage, args=[cpu_usage_before_calculation, stop_event])
        cpu_thread.daemon = True
        cpu_thread.start()

    def get_size_of_file(self, filepath, message=""):
        """Gets the file size of given path.

        Parameters
        ----------
        filepath : str
            filepath for function
        message : str, optional
             message for benchmark info, by default ""

        Returns
        -------
        int
            file size as bytes
        """
        size_of_file = 0
        if os.path.isdir(filepath):
            size_of_file += sum(os.stat(file).st_size for file in dataset_utils.get_targetted_files(filepath))
            dataset_utils.get_logger().info(f"{message} size of file {size_of_file} bytes")
        else:
            size_of_file = os.path.getsize(filepath)
            dataset_utils.get_logger().info(f"{message} size of file {size_of_file} bytes")
        return size_of_file
