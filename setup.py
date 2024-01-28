from setuptools import setup, find_packages

setup(
  name = 'pytorch-custom-utils',
  packages = find_packages(exclude=[]),
  version = '0.0.17',
  license='MIT',
  description = 'Pytorch Custom Utils',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/pytorch-custom-utils',
  keywords = [
    'pytorch',
    'accelerate'
  ],
  install_requires=[
    'accelerate',
    'optree',
    'pytorch-warmup',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
