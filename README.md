# Machine Learning for OpenCV

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/mbeyeler/opencv-machine-learning)

This is the Jupyter notebook version of the upcoming book [Machine Learning for OpenCV]() by Michael Beyeler.
The content is available on [GitHub](https://github.com/mbeyeler/opencv-machine-learning).

The code is released under the [MIT license](https://opensource.org/licenses/MIT).

## Table of Contents

[Preface](notebooks/00.00-Preface.ipynb)

1. [A Taste of Machine Learning](notebooks/01.00-A-Taste-of-Machine-Learning.ipynb)

2. [Working with Data in OpenCV](notebooks/02.00-Working-with-Data-in-OpenCV.ipynb)
   - [Dealing with Data Using Python's NumPy Package](notebooks/02.01-Dealing-with-Data-Using-Python-NumPy.ipynb)
   - [Loading External Datasets in Python](notebooks/02.02-Loading-External-Datasets-in-Python.ipynb)
   - [Visualizing Data Using Matplotlib](notebooks/02.03-Visualizing-Data-Using-Matplotlib.ipynb)
   - [Dealing with Data Using OpenCV's TrainData container](notebooks/02.05-Dealing-with-Data-Using-the-OpenCV-TrainData-Container-in-C%2B%2B.ipynb)

3. [First Steps in Supervised Learning](notebooks/03.00-First-Steps-in-Supervised-Learning.ipynb)
   - [The k-Nearest Neighbor Algorithm](notebooks/03.01-The-k-Nearest-Neighbor-Algorithm.ipynb)
   - [Logistic Regression](notebooks/03.02-Logistic-Regression.ipynb)
   - [Linear Regression](notebooks/03.03-Linear-Regression.ipynb)
   - [Ridge Regression](notebooks/03.04-Ridge-Regression.ipynb)

4. [Representing Data and Engineering Features](notebooks/04.00-Representing-Data-and-Engineering-Features.ipynb)
   - [Preprocessing Data](notebooks/04.01-Preprocessing-Data.ipynb)
   - [Reducing the Dimensionality of the Data](notebooks/04.02-Reducing-the-Dimensionality-of-the-Data.ipynb)
   - [Representing Categorical Variables](notebooks/04.03-Representing-Categorical-Variables.ipynb)
   - [Representing Text Features](notebooks/04.04-Represening-Text-Features.ipynb)
   - [Representing Images](notebooks/04.05-Representing-Images.ipynb)

5. [Using Decision Trees to Make a Medical Diagnosis](notebooks/05.00-Using-Decision-Trees-to-Make-a-Medical-Diagnosis.ipynb)
   - [Building Your First Decision Tree](notebooks/05.01-Building-Your-First-Decision-Tree.ipynb)
   - [Using Decision Trees to Diagnose Breast Cancer](notebooks/05.02-Using-Decision-Trees-to-Diagnose-Breast-Cancer.ipynb)
   - [Using Decision Trees for Regression](notebooks/05.03-Using-Decision-Trees-for-Regression.ipynb)

6. [Detecting Pedestrians with Support Vector Machines](notebooks/06.00-Detecting-Pedestrians-with-Support-Vector-Machines.ipynb)

7. [Implementing a Spam Filter with Bayesian Learning](notebooks/07.00-Implementing-a-Spam-Filter-with-Bayesian-Learning.ipynb)

8. [Discovering Hidden Structures with Unsupervised Learning](notebooks/08.00-Discovering-Hidden-Structures-with-Unsupervised-Learning.ipynb)

9. [Using Deep Learning to Classify Handwritten Digits](notebooks/09.00-Using-Deep-Learning-to-Classify-Handwritten-Digits.ipynb)

10. [Combining Different Algorithms Into an Ensemble](notebooks/10.00-Combining-Different-Algorithms-Into-an-Ensemble.ipynb)

11. [Selecting the Right Model with Hyper-Parameter Tuning](notebooks/11.00-Selecting-the-Right-Model-with-Hyper-Parameter-Tuning.ipynb)

12. [Conclusion](notebooks/12.00-Conclusion.ipynb)



## Running the Code

There are at least two ways you can run the code:
- by launching the project using [Binder](http://mybinder.org/repo/mbeyeler/opencv-machine-learning)
  (no installation required).
- from within a Jupyter notebook on your local machine.

The code in this book was tested with Python 3.5, although older versions of Python should work as well
(such as Python 2.7).


### Using Binder

[Binder](http://www.mybinder.org) allows you to run Jupyter notebooks in an interactive Docker container.
All you have to do is go to:
http://mybinder.org/repo/mbeyeler/opencv-machine-learning


### Using Jupyter Notebook

You basically want to follow the installation instructions in Chapter 1 of the book.

In short:

1. Download and install [Python Anaconda](https://www.continuum.io/downloads).

2. Create a conda environment for Python 3 with all required packages:

   ```
   $ conda create -n Python3 python=3.5 --file requirements.txt
   ```

3. Activate the conda environment.
   On Linux / Mac OS X:

   ```
   $ source activate Python3
   ```

   On Windows:

   ```
   $ activate Python3
   ```

   You can learn more about conda environments in the
   [Managing Environments](http://conda.pydata.org/docs/using/envs.html)
   section of the conda documentation.

4. Fork and clone the GitHub repo:
   - Click the
     [`Fork`](https://github.com/mbeyeler/opencv-machine-learning#fork-destination-box)
     button in the top-right corner of this page.
   - Clone the repo, where `YourUsername` is your actual GitHub user name:

     ```
     $ git clone https://github.com/YourUsername/opencv-machine-learning
     ```

5. Launch Jupyter notebook:

   ```
   $ jupyter notebook
   ```

   This will open up a browser window in your current directory.
   Navigate to `opencv-machine-learning/notebooks` and click on the notebook of your choice.
   Then select `Kernel > Restart & Run All`.
