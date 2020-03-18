# Classifying Dog Breeds

## Introduction

For an introduction to this project, please read the initial [README file](https://github.com/mfts/MLND-Dog-Breed-Classification-P3/blob/master/README.md). 

## Installation 

1. Clone the repository and navigate to the Notebooks folder.
	
	```	
	git clone https://github.com/mfts/MLND-Dog-Breed-Classification-P3.git
	cd MLND-Dog-Breed-Classification-P3/Notebooks
	```
	
2. Create a new environment with Python 3.6 and install packages & activate environment.

  ```
  conda env create -f environment.yml
  source activate dog-classify
  ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the nlpnd environment

  ```
  python -m ipykernel install --user --name dog-classify --display-name "dog-classify"
  ```

4. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.

  ```
  jupyter notebook dog_app.ipynb
  ```

Download datasets manually (optional):

5. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/Notebooks/data/dog_images`.  The `dog_images/` folder should contain 133 folders, each corresponding to a different dog breed.

6. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/Notebooks/data/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 