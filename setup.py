import re
from pathlib import Path
from typing import List, Tuple

from setuptools import find_packages, setup

NAME = "zerohertzLib"
DESCRIPTION = "Zerohertz's Library"
URL = ""
AUTHOR = "Zerohertz"
REQUIRES_PYTHON = ">=3.7.0"
HERE = Path(__file__).parent
try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


def get_requirements(
    path: str = HERE / "requirements.txt",
) -> Tuple[List[str], List[str]]:
    requirements = []
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip("\r\n")
            requirements.append(line)
    return requirements


def get_package_version() -> str:
    with open(HERE / f"{NAME}/__init__.py") as f:
        result = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if result:
            return result.group(1)
    raise RuntimeError("Can't get package version")


requirements = get_requirements()
version = get_package_version()


setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=requirements,
)
