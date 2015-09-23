from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_lib
import os.path

dbscan_dir = os.path.join('dbscan-v1.0.0', 'parallel_multicore')

dbscan_srcs = ['clusters.cpp', 'dbscan.cpp', 'kdtree2.cpp', 'utils.cpp']
dbscan_srcs = [os.path.join(dbscan_dir, src) for src in dbscan_srcs]

extra_dir = os.path.join('src')
extra_srcs = ['parallel_multicore_dbscan_module.cpp']
extra_srcs = [os.path.join(extra_dir, src) for src in extra_srcs]

numpy_include_dir = os.path.join(get_python_lib(), 'numpy', 'core', 'include')

dbscan_extension = Extension(
    name = 'parallel_dbscan._dbscan',
    sources = dbscan_srcs + extra_srcs,
    extra_compile_args = ['-fopenmp', '-std=c++11'],
    extra_link_args = ['-fopenmp', '-O3'],
    include_dirs = [dbscan_dir, numpy_include_dir]
)



packages = ['parallel_dbscan']
extensions = [dbscan_extension]

setup(name='parallel_dbscan',
      description='Parallel version of dbscan using OMP',
      version='0.1',
      packages=packages,
      ext_modules=extensions
      )
                            
