from distutils.core import setup, Extension
import numpy
import os

source_dir_root = "src"
omitted_sources = [
    "mod_xlsx.c",
    "mod_csv_reader.c",
    "readstat.c"
]
sources = [
    os.path.join(".", dirname, filename)
    for dirname, _, filenames in list(os.walk(source_dir_root))
    for filename in filenames
    if filename.endswith(".c") and
    dirname != "src/test" and
    filename not in omitted_sources]
print(sources)
source_dirs = [
    dirname for dirname, _, _ in list(os.walk(source_dir_root))
]
print(source_dirs)

smuggler = Extension(
    name='smuggler',
    library_dirs=source_dirs,
    libraries=["z"],
    sources=sources,
    define_macros=[("HAVE_XLSXWRITER", False)])

setup(
    name='Smuggler',
    version='0.1.1',
    description='Python binding to ReadStat library by Evan Miller.',
    include_dirs=[numpy.get_include()] + [source_dir_root],
    ext_modules=[smuggler],
    install_requires=["numpy", "pandas"])
