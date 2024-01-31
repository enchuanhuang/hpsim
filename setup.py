"""
Trying to install HPSim as a module with the following help:

cmdclass to compile HPSim.so
https://stackoverflow.com/questions/33168482/compiling-installing-c-executable-using-pythons-setuptools-setup-py
https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py
"""
import os, sys, glob
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from distutils.command.clean import clean
import subprocess
import pathlib
import shutil
PACKAGE_NAME = "hpsim"

class MakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class InstallMakeLibs(install_lib):
    """
    Get the libraries from the parent distribution, use those as the outfiles
    Skip building anything; everything is already built, forward libraries to
    the installation step
    """

    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """
        print("EC's InstallMakeLibs.run")
        self.announce("Moving library files")

        # We have already built the libraries in the previous build_ext step

        self.skip_build = True
        bin_dir = self.distribution.bin_dir

        # Depending on the files that are generated from your cmake
        # build chain, you may need to change the below code, such that
        # your files are moved to the appropriate location when the installation
        # is run

        libs = [os.path.join(bin_dir, _lib) for _lib in
                os.listdir(bin_dir) if
                os.path.isfile(os.path.join(bin_dir, _lib)) and
                os.path.splitext(_lib)[1] in [".dll", ".so"]
                and not (_lib.startswith("python") or _lib.startswith(PACKAGE_NAME))]

        print("copying libs")
        for lib in libs:
            print(f"copying {lib}")
            shutil.copy2(lib, os.path.join(self.build_dir,
                                          os.path.basename(lib)))

        # Mark the libs for installation, adding them to 
        # distribution.data_files seems to ensure that setuptools' record 
        # writer appends them to installed-files.txt in the package's egg-info
        #
        # Also tried adding the libraries to the distribution.libraries list, 
        # but that never seemed to add them to the installed-files.txt in the 
        # egg-info, and the online recommendation seems to be adding libraries 
        # into eager_resources in the call to setup(), which I think puts them 
        # in data_files anyways. 
        # 
        # What is the best way?

        # These are the additional installation files that should be
        # included in the package, but are resultant of the cmake build
        # step; depending on the files that are generated from your cmake
        # build chain, you may need to modify the below code

        self.distribution.data_files = [os.path.join(self.install_dir,
                                                     os.path.basename(lib))
                                        for lib in libs]
        # Must be forced to run after adding the libs to data_files
        print("install_data")
        self.distribution.run_command("install_data")




class BuildMakeExt(build_ext):
    """Builds using Makefile instead of the python implicit build"""

    def run(self):
        for extension in self.extensions:
            self.compile_and_install_software(extension)
        super().run()

    def compile_and_install_software(self, extension):
        """Used the subprocess module to compile/install the C software."""
        src_path = './src/'
        extension_path = pathlib.Path(self.get_ext_fullpath(extension.name))
        # compile the software
        cmd = "make"
        self.announce("Building C++/CUDA binary")
        subprocess.check_call(cmd, cwd=src_path, shell=True)

        self.announce("Moving built python library")
        bin_dir = os.path.dirname(os.path.realpath(__file__)) + "/bin/"
        bin_path = bin_dir + "HPSim.so"
        print(bin_path)
        try:
            shutil.copy2(bin_path, extension_path)
        except shutil.SameFileError:
            pass

        try:
            shutil.copy2(extension_path, "./")
        except shutil.SameFileError:
            pass

        # install the software (into the virtualenv bin dir if present)
        #subprocess.check_call('make install', cwd=src_path, shell=True)

class CleanExt(clean):
    def run(self):
        # Execute the classic clean command
        super().run()
        src_path = './src/'
        cmd = "make clean"
        print("Cleaning C++/CUDA binary")
        subprocess.check_call(cmd, cwd=src_path, shell=True)
        files = glob.glob("HPSim.cpython*.so")
        for file in files:
            print(f"removing file {file}")
            os.remove(file)
        for p in ["hpsim.egg-info", "bin", "build"]:
            if os.path.isdir(p):
                print(f"removing directory {p}")
                shutil.rmtree(p)
        print("Finish cleaning C++/CUDA binary")



setup(name='hpsim',
      version="1.1.0",
      author = "En-Chuan Huang, Larry Rybarycyk, Petr Anisimov",
      author_email = "en-chuan@lanl.gov",
      description=("High Performance Simulator"),
      keywords="particle accelerator LANSCE",
      install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'pyyaml',
        'tdqm'
      ],
      ext_modules=[MakeExtension("HPSim")],
      cmdclass={'build_ext': BuildMakeExt,
                "clean": CleanExt},
                #"install_lib": InstallMakeLibs},
      packages = ["hpsim"],
      # ...other settings skipped...
      )

