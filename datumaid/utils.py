from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import zipfile
from glob import glob
from logging.handlers import RotatingFileHandler

import cv2
import datumaro as dm
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_label_id(dataset, label_name):
    label_id = dataset.categories()[dm.AnnotationType.label].find(label_name)[0]
    if label_id is None:
        raise Exception("label id is None for label name " + label_name)
    return label_id


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def add_annotation_ids(dataset, labels_yml, tag):
    for label_yml in labels_yml:
        id = get_label_id(dataset, label_yml[tag])
        label_yml[tag + "_id"] = id


def generate_label_id_list(select_config):
    if "labels" not in select_config:
        return []
    return [label["label_id"] for label in select_config["labels"]]


def extract_zipfile(export_path, filepath):
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(export_path)


def generate_bbox_id_list(select_config):
    if "bboxes" not in select_config:
        return []
    return [label["bbox_id"] for label in select_config["bboxes"]]


def is_label(annotation):
    return type(annotation) is dm.components.annotation.Label


def is_bbox(annotation):
    return type(annotation) is dm.components.annotation.Bbox


def is_polygon(annotation):
    return type(annotation) is dm.components.annotation.Polygon


def check_annotation_has_config_attr(annotation_attrs, config_attrs):
    """Compare the config attributes with annotation attributes.
    config attrs has list supports.

    Parameters
    ----------
    annotation_attrs : Datumaro attribute item
        attributes of the annotion
    config_attrs : dictionary
        attributes comes from config yaml

    Returns
    -------
    Boolean
        True if annotation attrs has all of the config attrs. Else false
    """
    if config_attrs.items() > annotation_attrs.items():
        return False

    shared_items = {
        k: config_attrs[k]
        for k in config_attrs
        if k in annotation_attrs
        and (config_attrs[k] == annotation_attrs[k] or (hasattr(config_attrs[k], "__iter__") and annotation_attrs[k] in config_attrs[k]))
    }
    if shared_items == config_attrs:
        return True
    return False


def is_anotation_meet_item_yml(annotation, items_yml, item_type, item_id_tag):
    if item_type not in items_yml:
        return False
    for label in items_yml[item_type]:
        if annotation.label == label[item_id_tag]:
            if "attr" in label:
                if not annotation.attributes:
                    continue
                if check_annotation_has_config_attr(annotation.attributes, label["attr"]):
                    return True
            else:
                return True
    return False


def fix_image_path(item):
    if not os.path.exists(os.path.dirname(item.image.path)):
        return

    path = item.image.path

    if os.path.exists(path):
        return

    splitted_path = os.path.splitext(path)
    image_path_without_extension = splitted_path[0]

    def case_sensitive_path_search(image_existence_arr, extension):
        for img_path in image_existence_arr:
            if re.search(extension, img_path, re.IGNORECASE):
                return img_path

        raise Exception(f"The image folder does not contain default extension. image search path {image_path_without_extension}")

    image_existence = glob(image_path_without_extension + "*")

    if len(image_existence) == 1:
        item.image = dm.Image(path=image_existence[0])
    else:
        image_path = case_sensitive_path_search(image_existence, splitted_path[1])
        item.image = dm.Image(path=image_path)


def generate_mask(item):
    img_size = item.image.size
    mask_img = np.zeros(img_size, np.uint8)
    for annotation in item.annotations:
        if is_polygon(annotation):
            contours = [[int(x), int(y)] for x, y in zip(annotation.points[0::2], annotation.points[1::2])]
            mask_img = cv2.fillPoly(img=mask_img, pts=np.array([contours]), color=255)
        elif is_bbox(annotation):
            start_point = (int(annotation.x), int(annotation.y))
            end_point = (int(annotation.x + annotation.w), int(annotation.y + annotation.h))
            mask_img = cv2.rectangle(mask_img, start_point, end_point, 255, -1)
        else:
            pass

    return mask_img


def get_targetted_files(target_directory: str, target_extension_list=None, single_dept=False):
    """searches given target_directory recuresively and returns file paths
    if the file has an extension in given target_extension_list

    Parameters
    ----------
    dept : bool
        if toggled it will only fetch dept 1 files else it will bring all files in subfolders
    target_directory : str
        the directory to search in
    target_extension_list : list
        extension list to search if given None will return every file

    Returns
    -------
    list
        found files list
    """
    all_targetted_files_in_sub_directories = []
    target_extension_list = target_extension_list or [None]
    dept_level = "*" if single_dept else "**"
    for target in target_extension_list:
        temp = sorted(glob(target_directory + os.sep + dept_level + os.sep + "**" + (target or "*"), recursive=True), key=natural_keys)
        temp[:] = [item for item in temp if not os.path.isdir(item)]
        all_targetted_files_in_sub_directories.extend(temp)

    return all_targetted_files_in_sub_directories


def get_images(target_directory: str, single_dept=False):
    """searches the directory, returns the list of image paths

    Parameters
    ----------
    target_directory : str
        directory to search in

    Returns
    -------
    list
        found files list
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    ext_len = len(image_extensions)
    image_extensions[ext_len:] = [extension.upper() for extension in image_extensions]
    return get_targetted_files(target_directory, image_extensions, single_dept)


def merge_folders(target_images_list: list, destination_directory: str):
    """merges every directory to single directory

    Parameters
    ----------
    target_directory : str
        path to directory which contains directories to be merged
    destination_directory : str
        path to destination directory which all directories will be merged
        if there is not a valid directory it will be created
    """
    os.makedirs(destination_directory, exist_ok=True)
    [shutil.move(file, destination_directory) for file in target_images_list]


def create_video_from_frames(
    target_images_list: list, destination_directory: str, fourcc=cv2.VideoWriter_fourcc(*"MPEG"), fps=25, video_name="groundtruth.ts"
):
    """creates a video from target_directory's images

    Parameters
    ----------
    target_directory : str
        path to images which are going to be used for video
    destination_directory : str
        path of video
    fourcc : _type_, optional
        codec of video, by default cv2.VideoWriter_fourcc(*"MPEG")
    fps : int, optional
        frame rate per second of video, by default 25
    video_name : str, optional
        name of groundtruth video, by default "groundtruth.ts"
    """
    get_logger().info("Video creation started")
    output_video = destination_directory + os.sep + video_name
    video = cv2.VideoWriter(output_video, fourcc, fps, (1920, 1080))
    for image in tqdm(target_images_list, colour="green"):
        video.write(cv2.imread(image))

    video.release()
    get_logger().info(f"Video has been created and saved at {destination_directory}")


def get_perceptual_hash_is_same(image_1_path: str, image_2_path: str, sensitivity=1):
    """get perceptual hashes difference between two images

    Parameters
    ----------
    image_1_path : str
        path to image 1
    image_2_path : str
        path to image 2
    sensitivity : int
        hash size (2^(sensitivity+2))

    Returns
    -------
        is hash difference is 0
    """
    hash_size = 2 ** (2 + sensitivity)

    # Note that if image resized like %1 than changed to its original size hashes wont be same
    if os.path.exists(image_1_path) and os.path.exists(image_2_path):
        return imagehash.phash(Image.open(image_1_path), hash_size=hash_size) - imagehash.phash(Image.open(image_2_path), hash_size=hash_size) == 0
    else:
        missing_files = []
        if not os.path.exists(image_1_path):
            missing_files.append(image_1_path)
        if not os.path.exists(image_2_path):
            missing_files.append(image_2_path)
        get_logger().critical(f"Missing image(s) at: {missing_files}")
        raise FileNotFoundError(f"Missing image(s) at: {missing_files}")


def get_md5_hash_is_same(image_1_path: str, image_2_path: str, **kwargs):
    """get md5 hashes difference between two images

    Parameters
    ----------
    image_1_path : str
        path to image 1
    image_2_path : str
        path to image 2

    Returns
    -------
        is hash difference is 0
    """
    if os.path.exists(image_1_path) and os.path.exists(image_2_path):
        md5hash_1 = hashlib.md5(Image.open(image_1_path).tobytes())
        md5hash_2 = hashlib.md5(Image.open(image_2_path).tobytes())
        return md5hash_1.hexdigest() == md5hash_2.hexdigest()
    else:
        missing_files = []
        if not os.path.exists(image_1_path):
            missing_files.append(image_1_path)
        if not os.path.exists(image_2_path):
            missing_files.append(image_2_path)
        get_logger().critical(f"Missing image(s) at: {missing_files}")
        raise FileNotFoundError(f"Missing image(s) at: {missing_files}")


def get_duplicate_finder_parameters(target_directory: str, hash_type: str, sensitivity=1, duplicate_image_folder="duplicate_images"):
    """
      Returns the proper parameters for duplicate_finder function

    Parameters
    ----------
    target_directory : str
         specifies the directory that contains the files in which to find duplicates.
    hash_type : str
         specifies the hash type
    sensitivity : int, optional
         specifies the perceptual hash size (2^(sensitivity+2)), by default 1


    Returns
    -------
        parameters for duplicate_finder

    Raises
    ------
    Exception
        if hash_type is not "md5" or "perceptual"
    """

    if hash_type.lower() == "md5":
        if sensitivity > 5 and sensitivity <= 0 and hash_type.lower() == "md5":
            sensitivity = max(min(5, sensitivity), 1)
            get_logger().warning("Sensitivity can be between 1 and 5 ! Setting sensivity to {sensitivity}")

        def hash_function(image_path):
            if not os.path.exists(image_path):
                get_logger().critical(f"Missing image at: {image_path}")
                raise FileNotFoundError(f"Missing image at: {image_path}")
            return hashlib.md5(Image.open(image_path).tobytes()).hexdigest()

    elif hash_type.lower() == "perceptual":

        def hash_function(image_path):
            if not os.path.exists(image_path):
                get_logger().critical(f"Missing image at: {image_path}")
                raise FileNotFoundError(f"Missing image at: {image_path}")
            return int(str(imagehash.phash(Image.open(image_path), hash_size=2 ** (2 + sensitivity))), 16)

    else:
        get_logger().critical("Unrecognized hash function detected supported hash types are <perceptual,md5>")
        raise Exception(f"Invalid hash type selection. {hash_type} not found !")
    return target_directory, hash_function, duplicate_image_folder


def create_logger(**kwargs):
    """creates logger object

    Parameters
    ----------
    path : string
        logger file path
    level : logging enum, optional
        log level, by default logging.WARNING

    Returns
    -------
    logger
        logger object
    """
    if "log_path" not in kwargs:
        # path
        kwargs["log_path"] = "DatumAid.log"
    if "verbose" in kwargs:
        # level
        kwargs["log_level"] = logging.DEBUG
    else:
        # level
        kwargs["log_level"] = logging.INFO
    logger = logging.getLogger("DatumAid")
    # set level
    logger.setLevel(kwargs["log_level"])
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # add a rotating handler
    file_handler = RotatingFileHandler(kwargs["log_path"], maxBytes=20 * (1024**2), backupCount=1)  # 20MB
    file_handler.setFormatter(formatter)

    # add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # add handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


create_logger(kwargs={})


def get_logger():
    return logging.getLogger("DatumAid")
