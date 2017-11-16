import setuptools

setuptools.setup(
    name="tf_rmtpp",
    version="0.1.0",
    url="https://github.com/musically-ut/tf_rmtpp",

    author="Utkarsh Upadhyay",
    author_email="musically.ut@gmail.com",

    description="Recurrent Marked Temporal Point Processes in TensorFlow",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
