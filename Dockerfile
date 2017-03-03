FROM andrewosh/binder-base

MAINTAINER Andrew Osheroff <andrewosh@gmail.com>

USER main

# Add OpenCV 3.1
RUN conda install opencv=3.1