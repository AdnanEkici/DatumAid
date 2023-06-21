from __future__ import annotations

import csv
import multiprocessing
import os
import random
import shutil
import subprocess
import time
import unittest

import dask.dataframe as dd
import pandas as pd
import vaex
from parameterized import parameterized

import test.utils as test_utils

pc_benchmark_tools = test_utils.PCFunctionBenchmarkUtilities()


class BenchmarkUtilitiesTest(unittest.TestCase):
    test_parameters_for_cpu = [
        ("%100_cpu_usage", 100, 20),
        ("%50_cpu_usage", 50, 20),
        ("%25_cpu_usage", 25, 20),
    ]
    test_parameters_for_memory = [
        ("10gb_allocation", 1**10, 2500),
        ("1gb_allocation", 1**9, 2500),
        ("200mb_allocation", 2 * (10**8), 2500),
        ("1kb_allocation", 1000, 2500),
    ]
    test_parameters_for_access_time = [
        ("wait_0.01_seconds", 0.01),
        ("wait_0.1_seconds", 0.1),
        ("wait_0.5_seconds", 0.5),
        ("wait_1_seconds", 1),
        ("wait_10_seconds", 10),
    ]

    @parameterized.expand(test_parameters_for_cpu)
    def test_cpu_benchmark(self, _, usage, threshold):
        total_cpu = multiprocessing.cpu_count()

        @test_utils.PCFunctionBenchmarkUtilities.profile_cpu_usage
        def stress_cpu():
            subprocess.run(["stress", "-c", f"{int(total_cpu * usage / 100)}", "--timeout", "5s"])

        cpu_usage, _ = stress_cpu()  # noqa

        if not (cpu_usage > usage - threshold and cpu_usage < usage + threshold):
            self.fail(f"Expected cpu usage is {usage} mistake can be +{threshold} or -{threshold}. Calculated usage: {cpu_usage}")
        time.sleep(0.1)

    @parameterized.expand(test_parameters_for_memory)
    def test_memory_benchmark(self, _, memory_bytes, threshold):
        @test_utils.PCFunctionBenchmarkUtilities.profile_memory_usage
        def load_memory():
            tmp = bytearray(memory_bytes)
            del tmp

        memory_usage = load_memory()  # noqa
        memory_usage *= 1000000
        if not (memory_usage < memory_bytes + threshold and memory_usage > memory_bytes - threshold):
            self.fail(f"Expected memory usage is {memory_bytes}mb mistake can be +{threshold} or -{threshold}. Calculated usage: {memory_usage}")
        time.sleep(0.1)

    @parameterized.expand(test_parameters_for_access_time)
    def test_access_time_benchmark(self, _, seconds_to_wait):
        @test_utils.PCFunctionBenchmarkUtilities.profile_access_time
        def wait():
            time.sleep(seconds_to_wait)

        if seconds_to_wait == 0.01:
            self.skipTest(f"Acces time function does not have precision for {seconds_to_wait}")

        waited_time = wait()  # noqa
        self.assertEqual((round(waited_time, 1) - seconds_to_wait), 0, "Self difference between expected and actual should have been 0")
        time.sleep(0.1)


class DataframesBenchmarkTest(unittest.TestCase):
    path = os.sep + "tmp"
    testing_path = "dataframe_test_directory"
    test_enviorment_path = os.path.join(path, testing_path)
    csv_file_name = os.path.join(test_enviorment_path, "dummy_csv_file.csv")
    parquet_file_name = os.path.join(test_enviorment_path, "dummy_parquet_file.parquet")
    dask_saved_file_name = os.path.join(test_enviorment_path, "dask_saved_file_name.csv")
    feather_saved_file_name = os.path.join(test_enviorment_path, "dummy_feather_file.feather")

    test_parameters = [
        ("pandas_dataframe", csv_file_name, "Pandas dataframe", pd.read_csv, None),
        ("dask_dataframe", csv_file_name, "Dask dataframe", dd.read_csv, None),
        ("vaex_dataframe", parquet_file_name, "Vaex dataframe", vaex.open, None),
        ("feather_dataframe", feather_saved_file_name, "Feather dataframe", pd.read_feather, None),
        ("parquet_pyarrow", parquet_file_name, "Parquet(pyarrow engine)", pd.read_parquet, "pyarrow"),
        ("parquet_fastparquet", parquet_file_name, "Parquet(fastparquet engine)", pd.read_parquet, "fastparquet"),
    ]

    @classmethod
    def setUpClass(cls):
        if os.path.isdir(cls.test_enviorment_path):
            shutil.rmtree(cls.test_enviorment_path)
        os.makedirs(cls.test_enviorment_path, exist_ok=True)
        cols = 5
        rows = 1000  # optimal test scenario for benchmark is: 100000000
        data = [[random.randint(0, 100) for _ in range(cols)] for __ in range(rows)]

        with open(cls.csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            [writer.writerow(row) for row in data]

        csv_file = pd.read_csv(cls.csv_file_name)
        csv_file.to_parquet(cls.parquet_file_name)
        csv_file.to_feather(cls.feather_saved_file_name)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.test_enviorment_path):
            shutil.rmtree(cls.test_enviorment_path)

    @parameterized.expand(test_parameters)
    def test_memory_usage_time(self, _, filename, message, function, engine):
        @test_utils.PCFunctionBenchmarkUtilities.profile_memory_usage
        def wrapper(*args, message=message, **kwargs):
            function(*args, **kwargs)

        wrapper(filename, message=message, **({"engine": engine} if engine else {}))  # noqa

    @parameterized.expand(test_parameters)
    def test_access_time(self, _, filename, message, function, engine):
        @test_utils.PCFunctionBenchmarkUtilities.profile_access_time
        def wrapper(*args, message=message, **kwargs):
            function(*args, **kwargs)

        wrapper(filename, message=message, **({"engine": engine} if engine else {}))  # noqa

    @parameterized.expand(test_parameters)
    def test_cpu_usage(self, _, filename, message, function, engine):
        @test_utils.PCFunctionBenchmarkUtilities.profile_cpu_usage
        def wrapper(*args, message=message, **kwargs):
            function(*args, **kwargs)

        wrapper(filename, message=message, **({"engine": engine} if engine else {}))  # noqa
        time.sleep(1)

    @parameterized.expand([item for i, item in enumerate(test_parameters) if i != 0])
    def test_storage_of_data_frames(self, _, filepath, message, *args):
        pc_benchmark_tools.get_size_of_file(filepath=filepath, message=message)

    def step_1_save_times(self, filename, message, read_function, engine, save_function, save_file_name):
        dataframe = read_function(filename, **({"engine": engine} if engine else {}))

        @test_utils.PCFunctionBenchmarkUtilities.profile_access_time
        def save_wrapper(dataframe, save_file_name, message=message):
            save_function(dataframe, save_file_name)

        save_wrapper(dataframe, save_file_name, message=message)  # noqa

    def step_2_get_storage_sizes_of_saved_file(self, filepath: str, message=""):
        pc_benchmark_tools.get_size_of_file(filepath=filepath, message=message)

    @parameterized.expand(
        [
            (
                "save_pandas_dataframe_as_csv",  # Test name
                csv_file_name,  # File name to be read
                "Save pandas dataframe as csv",  # Log message
                pd.read_csv,  # Data read function
                None,  # Data read engine only specific to parquet
                lambda df, filename: df.to_csv(filename),  # dataframe save function
                # Save file or directory name
                os.path.join(test_enviorment_path, "saved_pandas_as_csv.csv"),
            ),
            (
                "save_pandas_dataframe_as_pickle",
                csv_file_name,
                "Save pandas dataframe as pickle",
                pd.read_csv,
                None,
                lambda df, filename: df.to_csv(filename),
                os.path.join(test_enviorment_path, "saved_pandas_as_pickle.pkl"),
            ),
            (
                "save_dask_dataframe_as_csv_multiple_file",
                csv_file_name,
                "Save Dask dataframe as csv (multiple file)",
                dd.read_csv,
                None,
                lambda df, filename: df.to_csv(filename, single_file=False),
                os.path.join(test_enviorment_path, "saved_dask_as_csv_multiple"),
            ),
            (
                "save_dask_dataframe_as_csv_single_file",
                csv_file_name,
                "Save Dask dataframe as csv (single file)",
                dd.read_csv,
                None,
                lambda df, filename: df.to_csv(filename, single_file=True),
                os.path.join(test_enviorment_path, "saved_dask_as_csv_single"),
            ),
            (
                "save_dask_dataframe_as_parquet",
                csv_file_name,
                "Save Dask dataframe as parquet",
                dd.read_csv,
                None,
                lambda df, filename: df.to_parquet(filename),
                os.path.join(test_enviorment_path, "saved_dask_as_parquet"),
            ),
            (
                "save_vaex_dataframe_as_parquet",
                parquet_file_name,
                "Save Vaex dataframe as parquet",
                vaex.open,
                None,
                lambda df, filename: df.export(filename),
                os.path.join(test_enviorment_path, "saved_vaex_as_parquet.parquet"),
            ),
            (
                "save_feather_dataframe",
                feather_saved_file_name,
                "Saved dataframe as feather",
                pd.read_feather,
                None,
                lambda df, filename: df.to_feather(filename),
                os.path.join(test_enviorment_path, "saved_feather.feather"),
            ),
            (
                "save_parquet_pyarrow",
                parquet_file_name,
                "Save Parquet(pyarrow engine)",
                pd.read_parquet,
                "pyarrow",
                lambda df, filename: df.to_parquet(filename),
                os.path.join(test_enviorment_path, "saved_pyarrow_parquet.parquet"),
            ),
            (
                "save_parquet_fastparquet",
                parquet_file_name,
                "Save Parquet(fastparquet engine)",
                pd.read_parquet,
                "fastparquet",
                lambda df, filename: df.to_parquet(filename),
                os.path.join(test_enviorment_path, "saved_fastparquet_parquet.parquet"),
            ),
        ]
    )
    def test_step_1_2_save_file_access_time_and_storage(self, _, filename, message, read_function, engine, save_function, save_file_name):
        self.step_1_save_times(filename, message, read_function, engine, save_function, save_file_name)
        self.step_2_get_storage_sizes_of_saved_file(filepath=save_file_name, message=os.path.basename(save_file_name))
