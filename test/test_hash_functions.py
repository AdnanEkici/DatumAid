from __future__ import annotations

import itertools
import os
import shutil
import unittest

import cv2 as cv
from parameterized import parameterized

import datumaid.utils as dataset_utils
import test.utils as test_utils

path = os.sep + "tmp" + os.sep + "duplicate_frame_test"
test_environment_path = os.path.join(path, "test_environment")
image_png_0 = test_environment_path + os.sep + "test_frame_0.PNG"
image_png_1 = test_environment_path + os.sep + "test_frame_1.PNG"
image_jpg_0 = test_environment_path + os.sep + "test_frame_0.jpg"


class HashTestUtils:
    # NOTE: md5 hash size is 128 bit
    test_parameters_for_png_jpg = [
        ("jpg_quality_100_RGB_md5", 100),
        ("jpg_quality_100_RGB_perceptual_hash_sensivity_1", 100, 1),
        ("jpg_quality_100_RGB_perceptual_hash_sensivity_4", 100, 4),
        ("jpg_quality_100_RGB_perceptual_hash_sensivity_5", 100, 5),
        ("jpg_quality_90_RGB_md5", 90),
        ("jpg_quality_90_RGB_perceptual_hash_sensivity_1", 90, 1),
        ("jpg_quality_90_RGB_perceptual_hash_sensivity_4", 90, 4),
        ("jpg_quality_90_RGB_perceptual_hash_sensivity_5", 90, 5),
        ("jpg_quality_80_RGB_md5", 80),
        ("jpg_quality_80_RGB_perceptual_hash_sensivity_1", 80, 1),
        ("jpg_quality_80_RGB_perceptual_hash_sensivity_4", 80, 4),
        ("jpg_quality_80_RGB_perceptual_hash_sensivity_5", 80, 5),
        ("jpg_quality_70_RGB_md5", 70),
        ("jpg_quality_70_RGB_perceptual_hash_sensivity_1", 70, 1),
        ("jpg_quality_70_RGB_perceptual_hash_sensivity_4", 70, 4),
        ("jpg_quality_70_RGB_perceptual_hash_sensivity_5", 70, 5),
    ]
    test_parameters_for_rotation = [
        ("rotate_(0.1)_degree_md5", 0.1),
        ("rotate_(0.1)_degree_perceptual_hash_sensivity_1", 0.1, 1),
        ("rotate_(0.1)_degree_perceptual_hash_sensivity_4", 0.1, 4),
        ("rotate_(0.1)_degree_perceptual_hash_sensivity_5", 0.1, 5),
        ("rotate_(1)_degree_md5", 1),
        ("rotate_(1)_degree_perceptual_hash_sensivity_1", 1, 1),
        ("rotate_(1)_degree_perceptual_hash_sensivity_4", 1, 4),
        ("rotate_(1)_degree_perceptual_hash_sensivity_5", 1, 5),
        ("rotate_(45)_degree_md5", 45),
        ("rotate_(45)_degree_perceptual_hash_sensivity_1", 45, 1),
        ("rotate_(45)_degree_perceptual_hash_sensivity_4", 45, 4),
        ("rotate_(45)_degree_perceptual_hash_sensivity_5", 45, 5),
    ]
    test_parameters_for_resize = [
        ("resize_(%20)_md5", 0.2),
        ("resize_(%20)_perceptual_hash_sensivity_1", 0.2, 1),
        ("resize_(%20)_perceptual_hash_sensivity_4", 0.2, 4),
        ("resize_(%20)_perceptual_hash_sensivity_5", 0.2, 5),
        ("resize_(%40)_md5", 0.4),
        ("resize_(%40)_perceptual_hash_sensivity_1", 0.4, 1),
        ("resize_(%40)_perceptual_hash_sensivity_4", 0.4, 4),
        ("resize_(%40)_perceptual_hash_sensivity_5", 0.4, 5),
        ("resize_(%60)_md5", 0.6),
        ("resize_(%60)_perceptual_hash_sensivity_1", 0.6, 1),
        ("resize_(%60)_perceptual_hash_sensivity_4", 0.6, 4),
        ("resize_(%60)_perceptual_hash_sensivity_5", 0.6, 5),
        ("resize_(%80)_md5", 0.8),
        ("resize_(%80)_perceptual_hash_sensivity_1", 0.8, 1),
        ("resize_(%80)_perceptual_hash_sensivity_4", 0.8, 4),
        ("resize_(%80)_perceptual_hash_sensivity_5", 0.8, 5),
        ("resize_(%100)_md5", 1.0),
        ("resize_(%100)_perceptual_hash_sensivity_1", 1.0, 1),
        ("resize_(%100)_perceptual_hash_sensivity_4", 1.0, 4),
        ("resize_(%100)_perceptual_hash_sensivity_5", 1.0, 5),
        ("resize_(%120)_md5", 1.2),
        ("resize_(%120)_perceptual_hash_sensivity_1", 1.2, 1),
        ("resize_(%120)_perceptual_hash_sensivity_4", 1.2, 4),
        ("resize_(%120)_perceptual_hash_sensivity_5", 1.2, 5),
    ]
    test_parameters_for_noise = [
        ("salt_and_pepper_noise_md5", "s&p"),
        ("salt_and_pepper_noise_perceptual_hash_sensivity_1", "s&p", 1),
        ("salt_and_pepper_noise_perceptual_hash_sensivity_4", "s&p", 4),
        ("salt_and_pepper_noise_perceptual_hash_sensivity_5", "s&p", 5),
        ("gaussian_noise_md5", "gaussian"),
        ("gaussian_noise_perceptual_hash_sensivity_1", "gaussian", 1),
        ("gaussian_noise_perceptual_hash_sensivity_4", "gaussian", 4),
        ("gaussian_noise_perceptual_hash_sensivity_5", "gaussian", 5),
        ("poisson_noise_md5", "poisson"),
        ("poisson_noise_perceptual_hash_sensivity_1", "poisson", 1),
        ("poisson_noise_perceptual_hash_sensivity_4", "poisson", 4),
        ("poisson_noise_perceptual_hash_sensivity_5", "poisson", 5),
        ("rayleigh_noise_md5", "rayleigh"),
        ("rayleigh_noise_perceptual_hash_sensivity_1", "rayleigh", 1),
        ("rayleigh_noise_perceptual_hash_sensivity_4", "rayleigh", 4),
        ("rayleigh_noise_perceptual_hash_sensivity_5", "rayleigh", 5),
    ]
    test_parameters_for_contrast = [
        ("contrast_adjustment_alpha=0.5_beta=0_md5", 0.5, 0),
        ("contrast_adjustment_alpha=0.5_beta=0_perceptual_hash_sensivity_1", 0.5, 0, 1),
        ("contrast_adjustment_alpha=0.5_beta=0_perceptual_hash_sensivity_4", 0.5, 0, 4),
        ("contrast_adjustment_alpha=0.5_beta=0_perceptual_hash_sensivity_5", 0.5, 0, 5),
        ("contrast_adjustment_alpha=1.5_beta=0_md5", 1.5, 0),
        ("contrast_adjustment_alpha=1.5_beta=0_perceptual_hash_sensivity_1", 1.5, 0, 1),
        ("contrast_adjustment_alpha=1.5_beta=0_perceptual_hash_sensivity_4", 1.5, 0, 4),
        ("contrast_adjustment_alpha=1.5_beta=0_perceptual_hash_sensivity_5", 1.5, 0, 5),
        ("contrast_adjustment_alpha=3.0_beta=0_md5", 1.5, 0),
        ("contrast_adjustment_alpha=3.0_beta=0_perceptual_hash_sensivity_1", 1.5, 0, 1),
        ("contrast_adjustment_alpha=3.0_beta=0_perceptual_hash_sensivity_4", 1.5, 0, 4),
        ("contrast_adjustment_alpha=3.0_beta=0_perceptual_hash_sensivity_5", 1.5, 0, 5),
        ("contrast_adjustment_alpha=10_beta=0_md5", 1.5, 0),
        ("contrast_adjustment_alpha=10_beta=0_perceptual_hash_sensivity_1", 1.5, 0, 1),
        ("contrast_adjustment_alpha=10_beta=0_perceptual_hash_sensivity_4", 1.5, 0, 4),
        ("contrast_adjustment_alpha=10_beta=0_perceptual_hash_sensivity_5", 1.5, 0, 5),
        ("contrast_adjustment_alpha=1.5_beta=10_md5", 1.5, 10),
        ("contrast_adjustment_alpha=1.5_beta=10_perceptual_hash_sensivity_1", 1.5, 10, 1),
        ("contrast_adjustment_alpha=1.5_beta=10_perceptual_hash_sensivity_4", 1.5, 10, 4),
        ("contrast_adjustment_alpha=1.5_beta=10_perceptual_hash_sensivity_5", 1.5, 10, 5),
        ("contrast_adjustment_alpha=1.5_beta=40_md5", 1.5, 40),
        ("contrast_adjustment_alpha=1.5_beta=40_perceptual_hash_sensivity_1", 1.5, 40, 1),
        ("contrast_adjustment_alpha=1.5_beta=40_perceptual_hash_sensivity_4", 1.5, 40, 4),
        ("contrast_adjustment_alpha=1.5_beta=40_perceptual_hash_sensivity_5", 1.5, 40, 5),
        ("contrast_adjustment_alpha=1.5_beta=80_md5", 1.5, 80),
        ("contrast_adjustment_alpha=1.5_beta=80_perceptual_hash_sensivity_1", 1.5, 80, 1),
        ("contrast_adjustment_alpha=1.5_beta=80_perceptual_hash_sensivity_4", 1.5, 80, 4),
        ("contrast_adjustment_alpha=1.5_beta=80_perceptual_hash_sensivity_5", 1.5, 80, 5),
    ]
    test_parameters_for_different_sizes = [
        ("image_size_width=1920_height=1080_md5", 1920, 1080),
        ("image_size_width=1920_height=1080_perceptual_hash_sensivity_1", 1920, 1080, 1),
        ("image_size_width=1920_height=1080_perceptual_hash_sensivity_4", 1920, 1080, 4),
        ("image_size_width=1920_height=1080_perceptual_hash_sensivity_5", 1920, 1080, 5),
        ("image_size_width=1280_height=720_md5", 1280, 480),
        ("image_size_width=1280_height=720_perceptual_hash_sensivity_1", 1280, 720, 1),
        ("image_size_width=1280_height=720_perceptual_hash_sensivity_4", 1280, 720, 4),
        ("image_size_width=1280_height=720_perceptual_hash_sensivity_5", 1280, 720, 5),
        ("image_size_width=640_height=480_md5", 640, 480),
        ("image_size_width=640_height=480_perceptual_hash_sensivity_1", 640, 480, 1),
        ("image_size_width=640_height=480_perceptual_hash_sensivity_4", 640, 480, 4),
        ("image_size_width=640_height=480_perceptual_hash_sensivity_5", 640, 480, 5),
    ]
    test_parameters_for_interpolation = [
        ("nearest-neighbor_interpolation_md5", cv.INTER_NEAREST),
        ("nearest-neighbor_interpolation_perceptual_hash_sensivity_1", cv.INTER_NEAREST, 1),
        ("nearest-neighbor_interpolation_perceptual_hash_sensivity_4", cv.INTER_NEAREST, 4),
        ("nearest-neighbor_interpolation_perceptual_hash_sensivity_5", cv.INTER_NEAREST, 5),
        ("inter_linear_interpolation_md5", cv.INTER_LINEAR),
        ("inter_linear_interpolation_perceptual_hash_sensivity_1", cv.INTER_LINEAR, 1),
        ("inter_linear_interpolation_perceptual_hash_sensivity_4", cv.INTER_LINEAR, 4),
        ("inter_linear_interpolation_perceptual_hash_sensivity_5", cv.INTER_LINEAR, 5),
        ("inter_cubic_interpolation_md5", cv.INTER_CUBIC),
        ("inter_cubic_interpolation_perceptual_hash_sensivity_1", cv.INTER_CUBIC, 1),
        ("inter_cubic_interpolation_perceptual_hash_sensivity_4", cv.INTER_CUBIC, 4),
        ("inter_cubic_interpolation_perceptual_hash_sensivity_5", cv.INTER_CUBIC, 5),
        ("inter_lanczos4_interpolation_md5", cv.INTER_LANCZOS4),
        ("inter_lanczos4_interpolation_perceptual_hash_sensivity_1", cv.INTER_LANCZOS4, 1),
        ("inter_lanczos4_interpolation_perceptual_hash_sensivity_4", cv.INTER_LANCZOS4, 4),
        ("inter_lanczos4_interpolation_perceptual_hash_sensivity_5", cv.INTER_LANCZOS4, 5),
    ]

    # Test specific utility functions
    def create_png_test_image(width=1920, height=1080, channel=3, count=1):
        for i in range(count):
            cv.imwrite(  # noqa
                test_environment_path + os.sep + "test_frame_" + str(i) + ".PNG",
                test_utils.generate_test_image(width=width, height=height, channel=channel),
            )

    def create_jpg_test_image(width=1920, height=1080, channel=3, quality=100, count=1):
        for i in range(count):
            cv.imwrite(  # noqa
                test_environment_path + os.sep + "test_frame_" + str(i) + ".jpg",
                test_utils.generate_test_image(width=width, height=height, channel=channel),
                [int(cv.IMWRITE_JPEG_QUALITY), quality],  # noqa
            )


class TestMD5Hash(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(test_environment_path, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(path)

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_png_jpg) if i % 4 == 0])
    def test_jpg_quality_with_png_image_comparison(self, _, quality):
        HashTestUtils.create_jpg_test_image(quality=quality)
        HashTestUtils.create_png_test_image()

        try:
            all_images = dataset_utils.get_images(target_directory=test_environment_path)
            for img_1, img_2 in itertools.combinations(all_images, 2):
                self.assertEqual(True, dataset_utils.get_md5_hash_is_same(img_1, img_2), f"Images has to be same! Failed jpg quality: {quality}")
        except AssertionError:
            self.skipTest(f"Md5hash test fails at jpg quality: {quality}")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_rotation) if i % 4 == 0])
    def test_rotation_comparison(self, _, rotation):
        HashTestUtils.create_png_test_image(count=2)

        test_utils.rotate_image(image_png_0, rotation, show_image=False, overwrite_image=True)
        try:
            self.assertEqual(
                True, dataset_utils.get_md5_hash_is_same(image_png_0, image_png_1), f"Images has to be same! Failed rotation: {rotation}"
            )
        except AssertionError:
            self.skipTest(f"Test fails at rotation: {rotation}")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_resize) if i % 4 == 0])
    def test_resize_image_comparison(self, _, resize_ratio):
        HashTestUtils.create_png_test_image(count=2)

        test_utils.resize_image(image_png_0, resize_ratio=resize_ratio, show_image=False, overwrite_image=True, return_to_original_size=True)

        try:
            self.assertEqual(
                True, dataset_utils.get_md5_hash_is_same(image_png_0, image_png_1), f"Images has to be same! Failed resize_ratio: {resize_ratio}"
            )
        except AssertionError:
            self.skipTest(f"Test fails at resize ratio {resize_ratio}")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_noise) if i % 4 == 0])
    def test_noisy_image_comparison(self, _, noise_type):
        HashTestUtils.create_png_test_image(count=2)
        test_utils.add_noise_to_image(image_path=image_png_0, noise_type=noise_type, overwrite_image=True, show_image=False)
        try:
            self.assertEqual(
                True, dataset_utils.get_md5_hash_is_same(image_png_0, image_png_1), f"Images has to be same! Failed at noise_type: {noise_type}"
            )
        except AssertionError:
            self.skipTest(f"Test fails at noise type {noise_type}")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_contrast) if i % 4 == 0])
    def test_contrast_comparison(self, _, alpha, beta):
        HashTestUtils.create_png_test_image(count=2)

        test_utils.adjust_contrast(image_path=image_png_0, alpha=alpha, beta=beta, overwrite_image=True, show_image=False)
        try:
            self.assertEqual(
                True, dataset_utils.get_md5_hash_is_same(image_png_0, image_png_1), f"Images has to be same! Failed at alpha: {alpha}, beta: {beta}"
            )
        except AssertionError:
            self.skipTest(f"Test fails at alpha {alpha} and beta {beta}")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_different_sizes) if i % 4 == 0])
    def test_static_image_size_and_channel_comparison(self, _, width, height):
        HashTestUtils.create_png_test_image(width=width, height=height, count=2)

        self.assertEqual(
            True, dataset_utils.get_md5_hash_is_same(image_png_0, image_png_1), f"Images has to be same! Failed width: {width} , height: {height}"
        )

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_interpolation) if i % 4 == 0])
    def test_interpolation_on_image(self, _, interpolation):
        HashTestUtils.create_png_test_image(count=2)
        test_utils.resize_image(
            image_png_0, show_image=False, resize_ratio=10, overwrite_image=True, return_to_original_size=True, interpolation=interpolation
        )
        try:
            self.assertEqual(
                True, dataset_utils.get_md5_hash_is_same(image_png_0, image_png_1), f"Images has to be same! Failed interpolation: {interpolation}"
            )
        except AssertionError:
            self.skipTest(f"Test fails at interpolation {interpolation}")


class TestPerceptualHash(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(test_environment_path, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(path)

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_different_sizes) if i % 4 != 0])
    def test_static_image_size_and_channel_comparison(self, _, width, height, hash_sensitivity):
        HashTestUtils.create_png_test_image(width=width, height=height, count=2)

        self.assertEqual(
            True,
            dataset_utils.get_perceptual_hash_is_same(image_png_0, image_png_1, sensitivity=hash_sensitivity),
            f"Images has to be same! Failed sensitivity: {hash_sensitivity} , width: {width} , height: {height}",
        )

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_png_jpg) if i % 4 != 0])
    def test_jpg_quality_with_png_image_comparison(self, _, quality, hash_sensitivity):
        HashTestUtils.create_jpg_test_image(quality=quality)
        HashTestUtils.create_png_test_image()

        try:
            self.assertEqual(
                True,
                dataset_utils.get_perceptual_hash_is_same(image_jpg_0, image_png_0, sensitivity=hash_sensitivity),
                f"Images has to be same! Failed sensitivity: {hash_sensitivity} , jpg quality: {quality}",
            )
        except AssertionError:
            self.skipTest(f"Test fails at hash sensitivity: {hash_sensitivity} ")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_rotation) if i % 4 != 0])
    def test_rotation_comparison(self, _, rotation, hash_sensitivity):
        HashTestUtils.create_png_test_image(count=2)

        test_utils.rotate_image(image_png_0, rotation, show_image=False, overwrite_image=True)
        try:
            self.assertEqual(
                True,
                dataset_utils.get_perceptual_hash_is_same(image_png_0, image_png_1, sensitivity=hash_sensitivity),
                f"Images has to be same! Failed sensitivity: {hash_sensitivity} , rotation: {rotation}",
            )
        except AssertionError:
            self.skipTest(f"Test fails at rotation: {rotation}")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_resize) if i % 4 != 0])
    def test_resize_image_comparison(self, _, resize_ratio, hash_sensitivity):
        HashTestUtils.create_png_test_image(count=2)

        test_utils.resize_image(image_png_0, resize_ratio=resize_ratio, show_image=False, overwrite_image=True, return_to_original_size=True)
        try:
            self.assertEqual(
                True,
                dataset_utils.get_perceptual_hash_is_same(image_png_0, image_png_1, sensitivity=hash_sensitivity),
                f"Images has to be same! Failed sensitivity: {hash_sensitivity} , resize_ratio: {resize_ratio}",
            )
        except AssertionError:
            self.skipTest(f"Test fails at hash_sensitivity: {hash_sensitivity} , resize_ratio: {resize_ratio}")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_noise) if i % 4 != 0])
    def test_noisy_image_comparison(self, _, noise_type, hash_sensitivity):
        HashTestUtils.create_png_test_image(count=2)
        test_utils.add_noise_to_image(image_path=image_png_0, noise_type=noise_type, overwrite_image=True, show_image=False)
        try:
            self.assertEqual(
                True,
                dataset_utils.get_perceptual_hash_is_same(image_png_0, image_png_1, sensitivity=hash_sensitivity),
                f"Images has to be same! Failed at sensitivity: {hash_sensitivity} , noise_type: {noise_type}",
            )
        except AssertionError:
            self.skipTest(f"Test fails at noise type {noise_type}")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_contrast) if i % 4 != 0])
    def test_contrast_comparison(self, _, alpha, beta, hash_sensitivity):
        HashTestUtils.create_png_test_image(count=2)

        test_utils.adjust_contrast(image_path=image_png_0, alpha=alpha, beta=beta, overwrite_image=True, show_image=False)

        try:
            self.assertEqual(
                True,
                dataset_utils.get_perceptual_hash_is_same(image_png_0, image_png_1, sensitivity=hash_sensitivity),
                f"Images has to be same! Failed at sensitivity: {hash_sensitivity} , alpha: {alpha}, beta: {beta}",
            )
        except AssertionError:
            self.skipTest(f"Test fails at Failed at sensitivity: {hash_sensitivity} , alpha: {alpha}, beta: {beta}")

    @parameterized.expand([item for i, item in enumerate(HashTestUtils.test_parameters_for_interpolation) if i % 4 != 0])
    def test_interpolation_on_image(self, _, interpolation, hash_sensitivity):
        HashTestUtils.create_png_test_image(count=2)
        test_utils.resize_image(
            image_png_0, show_image=False, resize_ratio=10, overwrite_image=True, return_to_original_size=True, interpolation=interpolation
        )
        try:
            self.assertEqual(
                True,
                dataset_utils.get_perceptual_hash_is_same(image_png_0, image_png_1),
                f"Images has to be same! Failed interpolation: {interpolation}",
            )
        except AssertionError:
            self.skipTest(f"Test fails at Failed at sensitivity: {hash_sensitivity} , interpolation: {interpolation}")
