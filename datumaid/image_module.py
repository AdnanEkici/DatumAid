from __future__ import annotations

import cv2
import numpy as np

import datumaid
from datumaid.utils import is_anotation_meet_item_yml


def fill_color(item, item_yml, color):
    """If annotation mathes with given item yml, fill the annotation with given color

    Parameters
    ----------
    item : Datumaro item
        item added by datumaro
    item_yml : item_yml
        the item part of the config_yml
    color : tuple of int
        color object like (255,255,255)
    """
    frame_cv = cv2.imread(item.image._path)

    is_frame_changed = False
    for annotation in item.annotations:
        if datumaid.utils.is_label(annotation):
            continue
        elif datumaid.utils.is_bbox(annotation):
            if is_anotation_meet_item_yml(annotation, item_yml, "bboxes", "bbox_id"):
                start_point = (int(annotation.x), int(annotation.y))
                end_point = (
                    int(annotation.x) + int(annotation.w),
                    int(annotation.y) + int(annotation.h),
                )
                cv2.rectangle(frame_cv, start_point, end_point, color, -1)
                is_frame_changed = True
        elif datumaid.utils.is_polygon(annotation):
            if is_anotation_meet_item_yml(annotation, item_yml, "polygons", "polygon_id"):
                iterator = iter(annotation.points)
                polygon_points = []
                try:
                    while True:
                        point1 = next(iterator)
                        point2 = next(iterator)
                        polygon_points.append([point1, point2])
                except StopIteration:
                    pass
                finally:
                    del iterator
                cv2.fillPoly(frame_cv, pts=np.int32([polygon_points]), color=color)
                is_frame_changed = True

    if is_frame_changed:
        cv2.imwrite(item.image._path, frame_cv)


def bitwise_not(item, item_yml):
    """If annotation matches with given item yml, apply bitwise not to the image for one time.

    Parameters
    ----------
    item : datumaro item
        item added by datumaro
    item_yml : item_yml
        the item part of the cofig yml
    """
    frame_cv = cv2.imread(item.image._path)

    change_frame = False
    for annotation in item.annotations:
        if datumaid.utils.is_label(annotation):
            if is_anotation_meet_item_yml(annotation, item_yml, "labels", "label_id"):
                change_frame = True
                break
        elif datumaid.utils.is_bbox(annotation):
            if is_anotation_meet_item_yml(annotation, item_yml, "bboxes", "bbox_id"):
                change_frame = True
                break
        elif datumaid.utils.is_polygon(annotation):
            if is_anotation_meet_item_yml(annotation, item_yml, "polygons", "polygon_id"):
                change_frame = True
                break

    if change_frame:
        frame_cv = cv2.bitwise_not(frame_cv)
        cv2.imwrite(item.image._path, frame_cv)
