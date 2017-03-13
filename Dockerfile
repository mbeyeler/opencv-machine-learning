FROM andrewosh/binder-base

MAINTAINER Michael Beyeler <mbeyeler@uw.edu> 

USER main

# Add OpenCV 3.1 for both the Py2 and Py3 kernels
RUN conda install opencv=3.1
RUN /bin/bash -c "source activate python3 && conda install -f opencv=3.1 && source deactivate"
