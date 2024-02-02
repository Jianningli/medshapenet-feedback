import MedShapeNetCore
from setuptools import setup, find_packages

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


README = readme()

 
setup(
    name='MedShapeNetCore',
    version=MedShapeNetCore.__version__,
    url='https://github.com/Jianningli/medshapenet-feedback/',
    license='Apache-2.0 License',
    author='Jianning Li',
    author_email='jianningli.me@gmail.com',
    python_requires=">=3.8.0",
    description='MedShapeNetCore: A Lightweight 3D Repository for Computer Vision and Machine Learning',
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pathlib",
        "scikit-learn",
        "scikit-image",
        "tqdm",
        "Pillow",
        "fire",
        "zenodo_get",
        "trimesh",
        "SimpleITK",
        "open3d",
        "scipy",
        "matplotlib",
        "clint",
        "pymeshfix",
        "requests",
        "argparse"
    ],
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only"
    ]
)


