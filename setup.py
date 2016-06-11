from distutils.core import setup, Extension
import numpy
import os

source_dir = "src"
sources = [
    os.path.join(source_dir, filename)
    for filename in os.listdir(source_dir)
    if filename.endswith(".c")]

smuggler = Extension(
    'smuggler',
    library_dirs=[source_dir],
    libraries=["z"],
    sources=sources)

setup(
    name='Smuggler',
    version='0.1',
    description='Python binding to ReadStat library by Evan Miller.',
    include_dirs=[numpy.get_include(), source_dir],
    ext_modules=[smuggler],
    install_requires=["numpy", "pandas"])
