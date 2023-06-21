from __future__ import annotations

import os.path as osp
import re

import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()


def find_version(project_dir=None):
    if not project_dir:
        project_dir = osp.dirname(osp.abspath(__file__))
    file_path = osp.join(project_dir, "datumaid", "version.py")
    with open(file_path) as version_file:
        version_text = version_file.read()
    # PEP440:
    # https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
    pep_regex = r"([1-9]\d*!)?(0|[1-9]\d*)(\.(0|[1-9]\d*))*((a|b|rc)(0|[1-9]\d*))?(\.post(0|[1-9]\d*))?(\.dev(0|[1-9]\d*))?"
    version_regex = r"VERSION\s*=\s*.(" + pep_regex + ")."
    match = re.match(version_regex, version_text)
    if not match:
        raise RuntimeError("Failed to find version string in '%s'" % file_path)

    version = version_text[match.start(1) : match.end(1)]  # noqa: E203
    return version


setuptools.setup(
    name="datumaid",
    version=find_version(),
    author="ordulu",
    description="Dataset Management Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ordulutechnology/DatumAid",
    project_urls={"Bug Tracker": "https://github.com/ordulutechnology/DatumAid/issues"},
    license="Ordulu License",
    packages=setuptools.find_packages(
        include=["datumaid"],
        exclude=["test"],
    ),
    install_requires=required,
    entry_points={
        "console_scripts": [
            "datumaid=datumaid.cli:main",
        ],
    },
)
