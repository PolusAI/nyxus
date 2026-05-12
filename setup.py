import os
import re
import sys
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_ver = out.decode().split("cmake version ")[-1].strip().split(".")
            if tuple(int(x) for x in cmake_ver[:2]) < (3, 1):
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DBUILD_LIB=ON", "-DALLEXTRAS=ON"
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        try:
            from setuptools_scm import get_version
            version = get_version(fallback_version="0.0.0")
        except Exception:
            version = "0.0.0"
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), version
        )

        if len(os.environ.get("CMAKE_ARGS", "")):
            cmake_args += os.environ.get("CMAKE_ARGS", "").split(" ")

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


def get_cuda_major_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        match = re.search(r'release (\d+)\.', output)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def get_name():
    if os.environ.get("NYXUS_GPU_WHEEL", "") == "ON":
        if len(os.environ.get("CMAKE_ARGS", "")):
            args = os.environ.get("CMAKE_ARGS", "").split(" ")
            if "-DUSEGPU=ON" in args:
                cuda_major_version = get_cuda_major_version()
                if cuda_major_version is None:
                    raise RuntimeError("USEGPU flag was set to ON but no CUDA version was found.")
                return f"nyxus-cuda{cuda_major_version}x"
    return "nyxus"


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name=get_name(),
    cmdclass={"build_ext": CMakeBuild},
    author="Andriy Kharchenko",
    author_email="andriy.kharchenko@axleinfo.com",
    url="https://github.com/PolusAI/nyxus",
    description="Scalable library for calculating features from intensity-label image data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("src/nyx/python"),
    package_dir={"": "src/nyx/python"},
    ext_modules=[CMakeExtension("nyxus/backend")],
    test_suite="tests",
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["numpy", "pandas"],
    tests_require=["pyarrow", "bfio"]
)
