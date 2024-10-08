from collections import defaultdict
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import sys
import numpy
import os
import os.path as path
from pathlib import Path

force = False
profile = False

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

compilation_includes = [".", numpy.get_include()]

setup_path = path.dirname(path.abspath(__file__))

# build extension list
extensions = []
for root, dirs, files in os.walk(setup_path):
    for file in files:
        if path.splitext(file)[1] == ".pyx":
            pyx_file = path.relpath(path.join(root, file), setup_path)
            module = path.splitext(pyx_file)[0].replace("/", ".")
            extensions.append(Extension(module, [pyx_file], include_dirs=compilation_includes),)

cython_directives = {"language_level": 3}
if profile:
    cython_directives["profile"] = True

# Include demos in a separate directory in the distribution as data_files.
demo_parent_path = Path("share/cherab/demos/imas")
data_files = defaultdict(list)
demos_source = Path("demos")
for item in demos_source.rglob("*"):
    if item.is_file():
        install_dir = demo_parent_path / item.parent.relative_to(demos_source)
        data_files[str(install_dir)].append(str(item))
data_files = list(data_files.items())

with open("README.md") as f:
    long_description = f.read()

setup(
    name="cherab-imas",
    version="0.1.1",
    namespace_packages=['cherab'],
    description="Cherab spectroscopy framework: IMAS submodule",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="https://github.com/vsnever/cherab-imas",
    project_urls=dict(
        Tracker="https://github.com/vsnever/cherab-imas/issues",
        Documentation="https://cherab.github.io/documentation/",
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={"": [
        "**/*.pyx", "**/*.pxd",  # Needed to build Cython extensions.
        ],
    },
    data_files=data_files,
    install_requires=["raysect==0.8.1.*", "cherab==1.5.*"],
    ext_modules=cythonize(extensions, force=force, compiler_directives=cython_directives),
)
