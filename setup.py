from setuptools import setup

setup(
    name='charart',
    version='0.1',
    description='Image & video character art converter',
    packages=['charart'],
    install_requires=['setuptools', 'numpy', 'moviepy', 'pillow', 'pytorch'],
)
