from __future__ import annotations

import itertools
import math
import os
import shutil
import unittest
from glob import glob

from parameterized import parameterized

import datumaid.utils as dataset_utils
from datumaid import DatasetManager

path_seperator = os.sep
path = os.path.join(os.sep, "tmp", "chunker_tests")
test_environment_path = os.path.join(path, "test_environment", "directory_to_be_chunked")
output_path = os.path.join(path, "chunked_output")

d_manager = DatasetManager(config_yml=None)

number_of_files_per_type = 7
max_file_type_per_dir = 3


class ChunkTests:
    @staticmethod
    def create_test_environment(enviroment_path, directory_names, number_of_files_per_type=300, target_list=None, non_target_file: bool = False):
        if non_target_file:
            target_list = target_list + [".JPG"]

        test_paths = [os.path.join(enviroment_path, directory) for directory in directory_names]
        for test_path in test_paths:
            os.makedirs(test_path, exist_ok=True)

        file_paths = [
            test_path + path_seperator + "frame_" + str(i).zfill(6) + target
            for target in target_list
            for i in range(number_of_files_per_type)
            for test_path in test_paths
        ]

        for file_path in file_paths:
            open(file_path, "w").close()

    @staticmethod
    def test_chunker_check_chunk_directories_names(self, _, target_list, enviroment, non_target_file, inplace):
        output_dir_path = test_environment_path if inplace else output_path
        number_of_chunks = math.ceil(number_of_files_per_type / max_file_type_per_dir)

        ChunkTests.create_test_environment(test_environment_path, enviroment, number_of_files_per_type, target_list, non_target_file)

        d_manager.chunker(
            target_directory=test_environment_path,
            target_extension_list=target_list,
            max_file_count=max_file_type_per_dir * len(target_list),
            chunked_output_directory_name=output_dir_path,
        )

        for directory in enviroment:
            chunk_dirs = sorted(glob(os.path.join(output_dir_path, directory, "*")), key=dataset_utils.natural_keys)
            chunk_dirs[:] = [chunk_dir for chunk_dir in chunk_dirs if os.path.isdir(chunk_dir)]
            self.assertEqual(
                len(chunk_dirs), number_of_chunks, f"Incorrect number of chunks. There should be {number_of_chunks} instead of {len(chunk_dirs)}"
            )
            assert_id_begin = 0
            assert_id_end = (assert_id_begin + max_file_type_per_dir) - 1
            for chunk_dir in chunk_dirs:
                check_id_begin, check_id_end = chunk_dir[chunk_dir.rfind(path_seperator) + 1 :].split("_")[0:2]  # noqa
                self.assertEqual(
                    (check_id_begin, check_id_end),
                    (str(assert_id_begin), str(assert_id_end)),
                    "Data has not been chunked properly. Expected chunk directory name should be "
                    f"{assert_id_begin}_{assert_id_end} instead of {check_id_begin}_{check_id_end}",
                )
                assert_id_begin += max_file_type_per_dir
                assert_id_end += max_file_type_per_dir

    @staticmethod
    def test_chunker_check_file_chunk_relation(self, _, target_list, enviroments, non_target_file, inplace):
        output_dir_path = test_environment_path if inplace else output_path

        ChunkTests.create_test_environment(test_environment_path, enviroments, number_of_files_per_type, target_list, non_target_file)

        d_manager.chunker(
            target_directory=test_environment_path,
            target_extension_list=target_list,
            max_file_count=max_file_type_per_dir * len(target_list),
            chunked_output_directory_name=output_dir_path,
        )

        chunk_dirs = list(itertools.chain(*[glob(os.path.join(output_dir_path, directory, "*")) for directory in enviroments]))
        chunk_dirs[:] = [chunk_dir for chunk_dir in chunk_dirs if os.path.isdir(chunk_dir)]

        for chunk in chunk_dirs:
            files_in_chunk = [file.split(".")[0].split("_")[1] for file in os.listdir(os.path.join(chunk))]
            files_min_id, files_max_id = min(files_in_chunk), max(files_in_chunk)
            assert_min, assert_max = chunk[chunk.rfind(path_seperator) + 1 :].split("_")[0:2]  # noqa
            self.assertTrue(int(assert_min) == int(files_min_id) <= int(files_max_id) <= int(assert_max), f"{chunk} contains wrong files")

    @staticmethod
    def test_chunker_check_safe_chunk(self, _, target_list, enviroments, non_target_file, inplace, safe_chunk):
        output_dir_path = test_environment_path if inplace else output_path

        ChunkTests.create_test_environment(test_environment_path, enviroments, number_of_files_per_type, target_list, non_target_file)

        files_before_chunk = set(itertools.chain(*[glob(os.path.join(test_environment_path, directory, "*")) for directory in enviroments]))
        non_target_files_before_chunk = {file for file in files_before_chunk if not any(file.endswith(target) for target in target_list)}

        d_manager.chunker(
            target_directory=test_environment_path,
            target_extension_list=target_list,
            max_file_count=max_file_type_per_dir * len(target_list),
            chunked_output_directory_name=output_dir_path,
            safe_chunk=safe_chunk,
        )

        files_after_chunk = {
            file
            for file in list(itertools.chain(*[glob(os.path.join(test_environment_path, directory, "*")) for directory in enviroments]))
            if os.path.isfile(file)
        }
        non_target_files_after_chunk = {file for file in files_after_chunk if not any(file.endswith(target) for target in target_list)}

        self.assertEqual(non_target_files_before_chunk, non_target_files_after_chunk, "Non target files should not be touched")
        if safe_chunk:
            self.assertEqual(files_before_chunk, files_after_chunk, "With safe chunk true, all the files in the target directory should not be moved")
        else:
            self.assertEqual(len(files_after_chunk), len(non_target_files_after_chunk), "With safe chunk false target files should be moved")

    @staticmethod
    def test_chunker_chunked(self, _, target_list, enviroments, non_target_file, inplace):
        output_dir_path = test_environment_path if inplace else output_path

        ChunkTests.create_test_environment(test_environment_path, enviroments, number_of_files_per_type, target_list, non_target_file)

        assertion_list = sorted(
            list(itertools.chain(*[glob(os.path.join(test_environment_path, directory, "*")) for directory in enviroments])),
            key=dataset_utils.natural_keys,
        )
        assertion_list[:] = [item for item in assertion_list if any(item.endswith(target) for target in target_list)]

        d_manager.chunker(
            target_directory=test_environment_path,
            target_extension_list=target_list,
            max_file_count=max_file_type_per_dir * len(target_list),
            chunked_output_directory_name=output_dir_path,
        )

        check_list = sorted(
            list(itertools.chain(*[glob(os.path.join(output_dir_path, directory, "*", "*")) for directory in enviroments])),
            key=dataset_utils.natural_keys,
        )

        self.assertTrue(
            all(
                check_list[i][check_list[i].rfind(path_seperator) + 1 :] == assertion_list[i][assertion_list[i].rfind(path_seperator) + 1 :]  # noqa
                for i in range(len(check_list))
            ),
            "Chunking has not been done properly",
        )

    @staticmethod
    def test_chunker_check_pairs(self, _, target_list, enviroments, non_target_file, inplace):
        output_dir_path = test_environment_path if inplace else output_path

        ChunkTests.create_test_environment(test_environment_path, enviroments, number_of_files_per_type, target_list, non_target_file)

        d_manager.chunker(
            target_directory=test_environment_path,
            target_extension_list=target_list,
            max_file_count=max_file_type_per_dir * len(target_list),
            chunked_output_directory_name=output_dir_path,
        )

        check_list = sorted(
            list(itertools.chain(*[glob(os.path.join(output_dir_path, directory, "*", "*")) for directory in enviroments])),
            key=dataset_utils.natural_keys,
        )

        target_list.sort()
        subsets = [check_list[index : index + len(target_list)] for index in range(0, len(check_list), len(target_list))]  # noqa

        self.assertTrue(
            all(all(file.endswith(extension) for file, extension in zip(subset, target_list)) for subset in subsets), "There is an unmatched pair"
        )


class TestChunkerInplace(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(path)

    test_parameters = [
        ("with_1_target_single_directory_inplace", [".PNG"], ["."], {"non_target_file": False, "inplace": True}),
        ("with_1_target_single_directory", [".PNG"], ["."], {"non_target_file": False, "inplace": False}),
    ]

    @parameterized.expand(test_parameters)
    def test_chunker_check_chunk_directories_names(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_chunk_directories_names(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand(test_parameters)
    def test_chunker_check_file_chunk_relation(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_file_chunk_relation(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand([[test[0], test[1], test[2], {**test[3], "safe_chunk": val}] for test in test_parameters for val in [True, False]])
    def test_chunker_check_safe_chunk(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_safe_chunk(
            self,
            _,
            target_list,
            enviroments,
            additional_information["non_target_file"],
            additional_information["inplace"],
            additional_information["safe_chunk"],
        )

    @parameterized.expand(test_parameters)
    def test_chunker_chunked(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_chunked(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand(test_parameters)
    def test_chunker_check_pairs(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_pairs(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )


class TestChunkerNonTargetFile(TestChunkerInplace):
    test_parameters = [
        ("with_1_target_single_directory_non_target_file", [".PNG"], ["."], {"non_target_file": True, "inplace": False}),
        ("with_1_target_single_directory", [".PNG"], ["."], {"non_target_file": False, "inplace": False}),
    ]

    @parameterized.expand(test_parameters)
    def test_chunker_check_chunk_directories_names(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_chunk_directories_names(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand(test_parameters)
    def test_chunker_check_file_chunk_relation(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_file_chunk_relation(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand([[test[0], test[1], test[2], {**test[3], "safe_chunk": val}] for test in test_parameters for val in [True, False]])
    def test_chunker_check_safe_chunk(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_safe_chunk(
            self,
            _,
            target_list,
            enviroments,
            additional_information["non_target_file"],
            additional_information["inplace"],
            additional_information["safe_chunk"],
        )

    @parameterized.expand(test_parameters)
    def test_chunker_chunked(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_chunked(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand(test_parameters)
    def test_chunker_check_pairs(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_pairs(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )


class TestChunkerEnviromentMultipleDirectory(TestChunkerNonTargetFile):
    test_parameters = [
        ("with_1_target_single_directory", [".PNG"], ["."], {"non_target_file": False, "inplace": False}),
        ("with_1_target_multiple_directory", [".PNG"], ["_1", "_2"], {"non_target_file": False, "inplace": False}),
        ("with_1_target_deep_single_directory", [".PNG"], ["_1/depth1/depth2/depth3"], {"non_target_file": False, "inplace": False}),
        (
            "with_1_target_deep_multiple_directory",
            [".PNG"],
            ["_1/depth1/depth2/depth3", "_2/depth1/depth2/depth3"],
            {"non_target_file": False, "inplace": False},
        ),
    ]

    @parameterized.expand(test_parameters)
    def test_chunker_check_chunk_directories_names(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_chunk_directories_names(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand(test_parameters)
    def test_chunker_check_file_chunk_relation(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_file_chunk_relation(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand([[test[0], test[1], test[2], {**test[3], "safe_chunk": val}] for test in test_parameters for val in [True, False]])
    def test_chunker_check_safe_chunk(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_safe_chunk(
            self,
            _,
            target_list,
            enviroments,
            additional_information["non_target_file"],
            additional_information["inplace"],
            additional_information["safe_chunk"],
        )

    @parameterized.expand(test_parameters)
    def test_chunker_chunked(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_chunked(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand(test_parameters)
    def test_chunker_check_pairs(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_pairs(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )


class TestChunkerTargetExtension(TestChunkerEnviromentMultipleDirectory):
    test_parameters = [
        ("with_1_target_single_directory", [".PNG"], ["."], {"non_target_file": False, "inplace": False}),
        ("with_2_target_single_directory", [".txt", ".PNG"], ["."], {"non_target_file": False, "inplace": False}),
        ("with_3_target_single_directory", ["_prev.PNG", "_target.PNG", ".PNG"], ["."], {"non_target_file": False, "inplace": False}),
    ]

    @parameterized.expand(test_parameters)
    def test_chunker_check_chunk_directories_names(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_chunk_directories_names(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand(test_parameters)
    def test_chunker_check_file_chunk_relation(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_file_chunk_relation(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand([[test[0], test[1], test[2], {**test[3], "safe_chunk": val}] for test in test_parameters for val in [True, False]])
    def test_chunker_check_safe_chunk(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_safe_chunk(
            self,
            _,
            target_list,
            enviroments,
            additional_information["non_target_file"],
            additional_information["inplace"],
            additional_information["safe_chunk"],
        )

    @parameterized.expand(test_parameters)
    def test_chunker_chunked(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_chunked(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )

    @parameterized.expand(test_parameters)
    def test_chunker_check_pairs(self, _, target_list, enviroments, additional_information):
        ChunkTests.test_chunker_check_pairs(
            self, _, target_list, enviroments, additional_information["non_target_file"], additional_information["inplace"]
        )
