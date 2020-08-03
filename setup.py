import os
from setuptools import setup


def read_requirements():
    """
    Parse requirements from requirements.txt.
    """
    reqs_path = os.path.join('backend', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

setup(
    name='audio_classifier',
    version='0.0.1',
    description='Content-based Audio Classification Tools',
    long_description=readme,
    author='Akihiro Inui',
    author_email='mail@akihiroinui.com',
    url='https://github.com/inuinana/audio_classification',
    license=license,
    install_requires=read_requirements(),
    )
