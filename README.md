# Non-binary-Evaluation-Food-Recommendation
This repository contains the Jupyter notebooks and source codes used for reproducing the results published in our Non-binary Evaluation of Next-basket Food Recommendation.

Please contact [Liu Yue](mailto:yueliu@smu.edu.sg?cc=liuyue715@hotmail.com) if you have any questions or problems.

## Requirements
The notebooks have been tested in Python 3.7 via Anaconda with the following packages:

* fpmc==0.0.0
* implicit
* lightfm
* rouge
* bert-score

See requirements.txt for a complete list.


## Pipeline

### Step 1: Performing recommendations
Run the notebook `1-6*.ipynb` to perform the recommendation tasks. Model, data and param folder: /data/yueliu/Recommendation @10.0.106.122


### Step 2: Exploratory data analysis
Run the notebooks `1-*.ipynb` to perform data analysis of repeat and novel consumption. The notebook requires data files from previous steps in the `data` folder.

__Outputs:__ Several reports will be generated and stored in the `figure` folder.

### Step 2: User Study 1 results
Run the notebooks `2-*.ipynb` to perform data analysis for User Study 1. The notebook requires data files from previous steps in the `data` folder.

__Outputs:__ Several reports will be generated and stored in the `data` folder.

### Step 3: User Study 2 results - substitution scores
Run the notebooks `3-*.ipynb` to perform data analysis for User Study 2. The notebook requires data files from previous steps in the `data` folder.

__Outputs:__ Several reports will be generated and stored in the `data` folder.

### Step 4: User Study 2 results - preference scores
Run the notebooks `4-*.ipynb` to perform data analysis for User Study 2. The notebook requires data files from previous steps in the `data` folder.

__Outputs:__ Several reports will be generated and stored in the `data` folder.



