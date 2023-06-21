# DatumAid

DatumAid is a datumaro based toolbox for dataset creation.

## Table of Contents


- [Features](#features)

- [Overview](#overview)

- [Installation](#installation)

- [Usage](#usage)

- [Known Issues](#known_issues)



## Overview

DatumAid seamlessly integrates with CVAT, enabling users
to download previously annotated data using CVAT’s API.
Users can create or delete labeling projects and automatically
download datasets in YAML format without any manual inter-
vention. DatumAid operates based on a YAML configuration,
allowing users to recreate the same dataset multiple times
and version datasets using YAML files. Moreover, DatumAid
supports automatic filtering based on user-defined criteria. An
annotated dataset may contain multiple scenarios, and users
may want to train their deep models on specific dataset types
(e.g., only nighttime weather conditions). DatumAid enables
users to filter the desired dataset based on specific labels or
tags. The filtered data can be utilized at any desired stage
in the configuration file. The tool supports logical operations
(AND, OR) in filtering processes, allowing users to select
labeled frames based on complex criteria (e.g., rainy and
snowy weather conditions).


## Features

* Cvat API
Using the API functions of cvat , users can download labeled tasks and projects from cvat by their respective IDs. It provides the option to choose the data format to be downloaded, such as COCO, CVAT, etc., and whether to download the images along with the data.

Yaml format for cvat api
```yaml

download:
url: http://x.x.x.x:port
token: token
project_id: [project_id_1, project_id_2] # Either project id or task id should be in both is not supported.
task_id: [task_id_1 , task_id_2]
exclude_tasks: [exclude_task_id_3 , exclude_task_id_4] # Only works for project_id_1
format: "Datumaro 1.0, CVAT for images 1.1 etc"
include_images: True / False
download_path: download_path

```

* Selection Module
The selection module allows filtering desired data from labeled data based on the type, name, or attributes of the labels. Additionally, the selection module can utilize logical operations such as "and" or "or" based on the order of execution.

Yaml format for selection module
```yaml

selects:
        - select:
          labels:
            - label: label_name
              attr:
                attribute: [attr]
          exec: selection_module.any_of_annotations(item, select_yml)

```

Yaml format for selection module (or logic)
```yaml

selects:
        - select:
          labels:
            - label: label_name
              attr:
                attribute: [attr]
            - label: corner_case
          exec: selection_module.any_of_annotations(item, select_yml)

```

Yaml format for selection module (and logic)
```yaml

    selects:
        - select:
          labels:
            - label: label_name
              attr:
                attribute: [attr]
          exec: selection_module.any_of_annotations(item, select_yml)
        - select:
          labels:
            - label: label_name_2
          exec: selection_module.any_of_annotations(item, select_yml)

```


* Maniplation Module
DatumAid also enables applying manipulation operations on labeled data. If desired, labeled portions can be colored, removed based on the label's type, name, attribute or can apply bitwise not.

Yaml format for maniplation module (remove label)
```yaml

item_manipulations:
      - item:
        label_type:
          - label_type: label_name
        exec: selection_module.remove_annotation(item, item_yml)

```

Yaml format for maniplation module (fill color)
```yaml

item_manipulations:
      - item:
        label_type:
          - label_type: label_name
        label_type:
          - label_type: label_name
        exec: image_module.fill_color(item, item_yml, color=125); selection_module.remove_annotation(item, item_yml)

```

Yaml format for maniplation module (biwise_not color)
```yaml

item_manipulations:
      - item:
        labels:
          - label: label_name
        exec: image_module.bitwise_not(item, item_yml)

```

*  Split Module
DatumAid also has the capability to divide a dataset into test, train, and validation sets based on the ratios provided by the user. This functionality is optional and can be used as per the user's requirement.

Yaml format for split module
```yaml

  split:
      subset:
        train: 0.6
        valid: 0.4

```


* Remap Module
DatumAid can remap labels in differently annotated datasets. It has the ability to assign new labels or redefine existing labels in order to create a consistent labeling scheme across the dataset. This feature allows for the reorganization and standardization of labels in various annotated datasets.

Yaml format for remap module
```yaml

    remap_labels:
      default: delete
      mapping:
        label_old: label_new
        label_name:

```

* Merge Module
DatumAid has the capability to merge multiple datasets, but it requires a common label scheme in order to do so. By having a shared set of labels, DatumAid can combine multiple datasets into a unified and cohesive dataset. This allows for the consolidation of diverse datasets into a single comprehensive dataset.

Yaml format for merge module
```yaml

    project_labels: [label_name_0, label_name_1, label_name_2 etc.]

```

* Export Module
DatumAid supports commonly used datasets in widespread usage. After performing all the necessary operations, DatumAid allows users to export the processed data into custom_dataset one that we have created or popular dataset formats (imagenet, coco, yolo etc.). This exported dataset can then be used to train deep learning models or for any other required purposes.

Yaml format for export module
```yaml

export:
  format: custom_dataset
  path: export_path
  type: csv / parquet # Exclusive only to custom_dataset

```

* Generator Module
The Generator module in DatumAid allows users to extract desired frames that occur before a target frame from a video loaded in cvat. This enables the generation of datasets for models that require multiple frames. Additionally, users have the option to use the Generator module instead of downloading individual images when downloading labeled tasks from cvat. Instead of downloading the images, users can use the video itself as input for the Generator module. It's important to note that currently the Generator module must work in conjunction with the custom_dataset functionality.

Yaml format for generator module
```yaml

  generator:
      cvat_data_path: path_of_cvat_data  # Optional: Provide the path to CVAT data.
      pollute_back: 4
      pollute_step: 4

```

* Chunker Module
When confronted with a large number of files, the func-
tioning of the operating system may be hindered, and opening
folders containing a high volume of files can potentially
decrease the lifespan of the system’s disk. To tackle this
issue, DatumAid offers users the flexibility to partition a folder
containing numerous files into multiple folders based on user
preferences. During the export process, users can specify the
maximum number of files per folder, leading to improved
organization and management of the dataset. Chunker can also be called individually.

Yaml format for chunker module
```yaml

chunk:
  target_directory: target_directory
  target_extension_list: [.extension1, .extension2, .extension3]
  max_file_count: 500
  chunked_output_directory_name: chunked_output_directory_name
  safe_chunk: True / False

```

* Duplicate Detector Module
To ensure that there are no duplicate data points in the dataset, users have the option to utilize MD5 hash and perceptual hash functions. By applying these functions to the exported visual data, users can optionally identify visually similar or identical images and choose to remove them from the dataset. This provides flexibility in handling duplicate data points based on the user's preference.

Yaml format for duplicate detector module
```yaml

check_duplicates:
  target_directory: target_folder
  hash_type: md5 / perceptual
  sensitivity: optional default is 5

```

## Installation

To install dataset manager first install requirments.

```bash
pip install -r requirements.txt
```

To be able to use CLI install DatumAid

```bash
pip install .
```

To ensure tests run properly install stress
```bash
sudo apt install -y stress
```

## Usage

* Show version
```bash
datumaid version
```
* Process Dataset
```bash
datumaid dataset process --config-path config.yml
```
* Execute Chunker
```bash
datumaid dataset chunk --config-path config.yml
```
* Execute Detect Duplicate Frames
```bash
datumaid dataset find_duplicate --config-path config.yml
```
## Known Issues


UNDER CONSTRUCTION
