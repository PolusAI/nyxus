import os
import re
import sys
import versioneer
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        print("-----------------__init__ (" + str(Extension) + ")")
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        print("-----------------__init__ self.sourcedir=" + str(self.sourcedir))


class CMakeBuild(build_ext):
    def run(self):
        print("-----------------CMakeBuild::run()...")

        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print("-----------------CMakeBuild::build_extension()...")

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DBUILD_LIB=ON", "-DALLEXTRAS=ON"
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            print("-----------------Windows...")
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
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), versioneer.get_version()
        )

        if len(os.environ.get("CMAKE_ARGS", "")):
            cmake_args += os.environ.get("CMAKE_ARGS", "").split(" ")

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print("--------------- cmake_args=" + str(cmake_args))
        print("--------------- build_args=" + str(build_args))

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        print()  # Add an empty line for cleaner output

    
def get_cuda_major_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        match = re.search(r'release (\d+)\.', output)
        if match:
            return match.group(1)
        else:
            return None
    except Exception as e:
        return None    
    
def get_name():
    if os.environ.get("NYXUS_GPU_WHEEL", "") == "ON":
        if len(os.environ.get("CMAKE_ARGS", "")):
            args = os.environ.get("CMAKE_ARGS", "").split(" ")
            if "-DUSEGPU=ON" in args: #check if gpu build is requested
                cuda_major_version = get_cuda_major_version()
                if cuda_major_version is None:
                    raise RuntimeError("USEGPU flag was set to ON but no CUDA version was found. Set USEGPU=OFF to continue.")
                else:
                    return f"nyxus-cuda{cuda_major_version}x"
    return "nyxus"


with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
setup(
    name=get_name(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(dict(build_ext=CMakeBuild)),
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
    tests_require = ["pyarrow", "bfio"]
)
