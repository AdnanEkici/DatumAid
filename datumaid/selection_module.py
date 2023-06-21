from __future__ import annotations

import datumaid
from datumaid.utils import is_anotation_meet_item_yml


def any_of_annotations(item, select_yml):
    """Selects item annotations, if any of the annotation in config
    found in the items.

    Parameters
    ----------
    item : Datumaro item
        The item which is added by datumaro
    select_yml : select_yaml
        the select part of the config_yaml

    Returns
    -------
    Boolean
        True if item is selected, False if not
    """

    for annotation in item.annotations:
        if datumaid.utils.is_label(annotation):
            if is_anotation_meet_item_yml(annotation, select_yml, "labels", "label_id"):
                return True
        elif datumaid.utils.is_bbox(annotation):
            if is_anotation_meet_item_yml(annotation, select_yml, "bboxes", "bbox_id"):
                return True
        elif datumaid.utils.is_polygon(annotation):
            if is_anotation_meet_item_yml(annotation, select_yml, "polygons", "polygon_id"):
                return True

    return False


def remove_annotation(item, item_yml):
    """Removes annotations if annotaions mathes with item_yml

    Parameters
    ----------
    item : Datumaro item
        The item whish is added by datumaro
    item_yml : item_yml
        the item part of the config_yml
    """
    annotation_to_remove_list = []
    for annotation in item.annotations:
        if datumaid.utils.is_label(annotation):
            if is_anotation_meet_item_yml(annotation, item_yml, "labels", "label_id"):
                annotation_to_remove_list.append(annotation)
        elif datumaid.utils.is_bbox(annotation):
            if is_anotation_meet_item_yml(annotation, item_yml, "bboxes", "bbox_id"):
                annotation_to_remove_list.append(annotation)
        elif datumaid.utils.is_polygon(annotation):
            if is_anotation_meet_item_yml(annotation, item_yml, "polygons", "polygon_id"):
                annotation_to_remove_list.append(annotation)

    for annotation_to_remove in annotation_to_remove_list:
        item.annotations.remove(annotation_to_remove)
