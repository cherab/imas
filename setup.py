import sys
from collections import defaultdict
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

force = False
profile = False

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

compilation_includes = [".", numpy.get_include()]

setup_path = Path(__file__).parent

# build extension list
extensions = []
for pyx in (setup_path / "cherab").glob("**/*.pyx"):
    pyx_path = pyx.relative_to(setup_path)
    module = ".".join(pyx_path.with_suffix("").parts)
    extensions.append(
        Extension(module, [str(pyx_path)], include_dirs=compilation_includes),
    )

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
    namespace_packages=["cherab"],
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
