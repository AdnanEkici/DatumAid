from __future__ import annotations

import concurrent.futures
import json
import os
from functools import partial
from time import sleep

import requests
from requests.compat import quote_plus
from requests.compat import urljoin
from tqdm import tqdm

import datumaid.utils as utils


class CVATHandler:
    class MissingParameterError(Exception):
        def __init__(self, parameter_name):
            self.message = f"Missing parameter: {self.parameter_name}"

    class HttpError(Exception):
        pass

    def __init__(self, host_url, api_token):
        self.__headers = {"Authorization": f"Token {api_token}"}
        self.server_url = host_url
        self.__project_id = None
        self.__organization = None

    def __handle_request(self, req_type, url, **kwargs):
        """Performs HTTP requests with given parameters. Raises exception
        if any HTTP error occures.

        Parameters
        ----------
        req_type : str
            Type of the request eg. GET,POST
        url : str
            Server URL where the request send to.

        Returns
        -------
        class:`Response <Response>` object

        Raises
        ------
        CVATHandler.HttpError
            Raises exception if any HTTP error occures.
        """
        try:
            r = requests.request(req_type, url=url, **kwargs)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise CVATHandler.HttpError(e)

        return r

    def create_project(self, name, labels, organization):
        """Creates new CVAT project and sets the project id of it, if this function called
        the class instance uses that project id regardles of if any other provided.

        Parameters
        ----------
        name : str
            Name of the project.
        labels : list
            List of dictionaries includes labels of the project.
        organizatin: str
            Organization name.

        Returns
        -------
        str: Id of the created project
        """
        url = urljoin(self.server_url, f"api/projects?org={organization}")
        data = json.dumps({"name": name, "labels": labels})
        r = self.__handle_request("POST", url=url, headers={**self.__headers, "Content-Type": "application/json"}, data=data)

        self.__project_id = json.loads(r.text)["id"]
        self.__organization = organization

        return self.__project_id

    def delete_project(self, id=None):
        """Deletes CVAT project. If a project has not been created
        by using `create_project` function, the `id` argument is used.

        Parameters
        ----------
        id : int, optional
            Id of the project, by default None
        """
        url = urljoin(self.server_url, f"api/projects/{self.__project_id or id}")
        _ = self.__handle_request("DELETE", url=url, headers=self.__headers)

    def create_task(self, name, organization=None, subset="", project_id=None, labels=None, segment_size=200):
        """Creates a CVAT task. Uses `project_id` and `organization` initilized
        by `create_project` function if not provided.

        Parameters
        ----------
        name : str
            Name of the task
        subset : str, optional
            Subset of the task; train, test or validation, by default ""
        project_id : int, optional
            Id of the project the task will assign to, by default None
        labels : list, optional
            List of dictionaries includes labels of the task, by default []
        segment_size : int, optional
            Segment size of the jobs, by default 200
        organizatin: str, optional
            Organization name, by default None

        Returns
        -------
        int
            Id of the created task
        """
        if labels is None:
            labels = []
        if not self.__project_id and not project_id:
            raise CVATHandler.MissingParameterError("project_id")
        if not self.__organization and not organization:
            raise CVATHandler.MissingParameterError("organization")

        url = urljoin(self.server_url, f"api/tasks?org={organization or self.__organization}")
        data = json.dumps(
            {
                "name": name,
                "labels": labels,
                "project_id": project_id or self.__project_id,
                "overlap": 0,
                "subset": str.capitalize(subset),
                "segment_size": segment_size,
                "target_storage": {"location": "local"},
                "source_storage": {"location": "local"},
            }
        )
        r = self.__handle_request("POST", url=url, headers={**self.__headers, "Content-Type": "application/json"}, data=data)
        return json.loads(r.text)["id"]

    def delete_task(self, task_id):
        """Deletes a CVAT task.

        Parameters
        ----------
        task_id : int
            Id of the task
        """
        url = urljoin(self.server_url, f"api/tasks/{task_id}")
        _ = self.__handle_request("DELETE", url=url, headers=self.__headers)

    def upload_files(self, task_id, file_path, image_quality=90, step=1, is_video=False):
        """Uploads image or video data to the specified task. In case of `is_video` is False
        `file_path` must be directory path of the image files, otherwise -in case of video upload-
        `file_path` must be path of the video file.

        Parameters
        ----------
        task_id : int
            Id of the task
        file_path : str
            Directory of image files or path of the video
        image_quality : int, optional
            Quality value of the images in CVAT, between 0-100, by default 90
        step : int, optional
            Step size, by default 1
        is_video : bool, optional
            Indicator of if the file is a video, by default False
        """
        url = urljoin(self.server_url, f"api/tasks/{task_id}/data?org={self.__organization}")
        data = {
            "image_quality": image_quality,
            "frame_filter": f"step={step}",
            "use_zip_chunks": True,
            "use_cache": True,
            "sorting_method": "lexicographical",
        }

        headers = {**self.__headers, "Upload-Start": "true"}
        _ = self.__handle_request("POST", url=url, headers=headers, data=data)

        file_list = [file_path] if is_video else utils.get_images(file_path)
        files = {f"client_files[{i}]": open(f, "rb") for i, f in enumerate(file_list)}
        headers = {**self.__headers, "Upload-Multiple": "true"}
        _ = self.__handle_request("POST", url=url, headers=headers, data=data, files=files)

        data["client_files"] = []
        headers = {**self.__headers, "Upload-Finish": "true"}
        _ = self.__handle_request("POST", url=url, headers=headers, data=data)

        status_url = urljoin(self.server_url, f"api/tasks/{task_id}/status?org={self.__organization}")
        progress = "Started"
        while progress == "Started":
            r = self.__handle_request("GET", url=status_url, headers=self.__headers)
            progress = json.loads(r.text)["state"]
            if progress == "Finished":
                utils.get_logger().info(f"Task {task_id}: Upload completed...")
                break
            sleep(4)

    def upload_annotations(self, task_id, file_path, format="CVAT 1.1"):
        """Uploads annotations to the specified task.

        Parameters
        ----------
        task_id : int
            Id of the task
        file_path : str
            Path of the annotation file
        format : str, optional
            Annotation format,
            visit `api/server/annotation/formats` endpoint on your CVAT server for more information,
            by default "CVAT 1.1"
        """
        url = urljoin(self.server_url, f"api/tasks/{task_id}/annotations/?format={quote_plus(format)}")

        files = {"annotation_file": open(file_path, "rb")}
        r = self.__handle_request("PUT", url=url, headers=self.__headers, files=files)
        while r.status_code == 202:
            r = self.__handle_request(
                "PUT",
                url=url,
                headers=self.__headers,
            )
            if r.status_code == 201:
                utils.get_logger().info(f"Task {task_id}: Upload annotation finished!")
            sleep(1)

    def export_task(self, task_id, export_type, export_path, format):
        """Downloads the specified task.

        Parameters
        ----------
        task_id : int
            Id of the task
        export_type : str
            Indicator of if images going to be downloaded.
            Use `annotations` to download only annotations,
            `dataset` to download with images.
        export_path : str
            Path where the task going to be stored
        format : str
            Format of the annotation.
            Visit `api/server/annotation/formats` endpoint on your CVAT server for more information

        Returns
        -------
        str
            Filename of the downloaded task
        """
        filename = f"{export_type}_{task_id}.zip"
        url = urljoin(
            self.server_url,
            f"api/tasks/{task_id}/{export_type}?action=download&filename={filename}&format={quote_plus(format)}",
        )
        r = self.__handle_request("GET", url=url, headers=self.__headers, stream=True)

        content_length = int(r.headers.get("content-length", 0))

        while r.status_code != 200:
            r = self.__handle_request("GET", url=url, headers=self.__headers, stream=True)
            if r.status_code == 200:
                content_length = int(r.headers.get("content-length", 0))
                if content_length and content_length != 0:
                    break
                else:
                    utils.get_logger().warning(f"Something went wrong when exporting task {task_id}: Content has no data")
                    raise ValueError(f"Unexpected content \nvalue:{content_length}\ntype: {type(content_length)}")
            sleep(4)

        os.makedirs(export_path, exist_ok=True)
        block_size = 1024
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Task {task_id}:")
        with open(os.path.join(export_path, filename), "wb") as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()

        return str(os.path.join(export_path, filename))

    def get_task_list(self, project_id):
        """Gets task list of given project id.

        Parameters
        ----------
        project_id : int
          Unique project id in cvat

        Returns
        -------
        list
            task list of project
        """
        url = urljoin(self.server_url, f"api/projects/{project_id}/tasks?page_size=999")
        r = self.__handle_request("GET", url=url, headers=self.__headers)
        return [i["id"] for i in json.loads(r.content)["results"]]

    def get_export_type(self, include_images: bool):
        """Returns projects export type whether it will be downloaded with images or not

        Parameters
        ----------
        include_images : bool
            flag that decides will export have images or not

        Returns
        -------
        str
            dataset keyword for images, annotations for without images.
        """
        if include_images:
            return "dataset"
        return "annotations"

    def export_project(self, project_id, export_type, export_path, format, exclude_list=None):
        """Export all tasks in given cvat project id.

        Parameters
        ----------
        project_id : int
            unique project id in cvat
        export_type : str
            keyword that decides will export have images or not <dataset / annotations>
        export_path : str
            path where will export will be downloaded.
        format : str
            keyword that decides which format will dataset be downloaded <CVAT for images 1.1 / Datumaro 1.0>.
            More formats can be found http://localhost:8080/api/server/annotation/formats.
        exclude_list : list, optional
            List that excludes task ids in project that will not be downloaded , by default None

        Returns
        -------
        list
            list of paths where annotations are.
        """
        if exclude_list is None:
            exclude_list = []
        export_list = list(set(self.get_task_list(project_id=project_id)) - set(exclude_list))
        func = partial(self.export_task, export_type=export_type, export_path=export_path, format=format)
        cpu_count = os.cpu_count() // 2 if os.cpu_count() > 1 else 1  # Give half of the cpu.
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(func, task_id) for task_id in export_list]
            if any(future.exception() is not None for future in concurrent.futures.as_completed(futures)):
                utils.get_logger().critical(
                    "\n".join(
                        [
                            f"Task {os.path.basename(future.result())} failed with exception: {future.exception()}"
                            for future in concurrent.futures.as_completed(futures)
                            if future.exception() is not None
                        ]
                    )
                )

        return [future.result() for future in futures]
