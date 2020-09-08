import os
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

version = '0.1.0a0'
sha = 'Unknown'
package_name = 'cuda_ext'

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, package_name, 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))


write_version_file()

with open('README.md') as f:
    readme = f.read()


requirements = [
    'torch>=1.2',
]


setup(
    name=package_name,
    version=version,
    author='François-Guillaume Fernandez',
    description='Sample C++/CUDA extension for PyTorch',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/frgfm/torch-cuda-template',
    download_url='https://github.com/frgfm/torch-cuda-template/tags',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords=['pytorch', 'cpp', 'cuda', 'deep learning'],
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=requirements,
    ext_modules=[
          CUDAExtension(f"{package_name}._C", [
              'csrc/dsigmoid.cpp',
              'csrc/dsigmoid_kernel.cu',
              ],
              extra_compile_args={
                'cxx': [],
                'nvcc': ['--expt-extended-lambda']
              },
              include_dirs=['external']
          )
      ],
    # Disabling ninja for now
    # cf. https://github.com/microsoft/DeepSpeed/issues/280
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
    package_data={'': ['LICENSE']},
)
