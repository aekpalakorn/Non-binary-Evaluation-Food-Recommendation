# Non-binary Evaluation of Next-basket Food Recommendation

This repository contains the Jupyter notebooks and source codes used for reproducing the results published in our [paper](https://link.springer.com/article/10.1007/s11257-023-09369-8), **"Non-binary Evaluation of Next-basket Food Recommendation"** (User Modeling and User-Adapted Interaction, 2024).

Please contact [Liu Yue](mailto:yueliu@smu.edu.sg?cc=liuyue715@hotmail.com) if you have any questions or problems.

### Important Note on Data and Model Files

* The raw food consumption data used in the study is based on the anonymized MyFitnessPal public food diary [dataset](https://www.researchgate.net/publication/324601551_MyFitnessPal_Food_Diary_Dataset) (Weber and Achananuparp 2016). The specific processed, structured, and cleaned data files used directly in the notebooks are not included in this repository. Users must prepare their own version of this dataset for full replication.
* The package `fpmc` refers to a custom implementation of the Factorizing Personalized Markov Chains model adapted for this work.
* Notebooks assume the raw/processed data files are placed in a **`/data/`** directory (or a mock dataset structured similarly to the original).

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
Run the notebook `1-6*.ipynb` to perform the recommendation tasks.


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

## Citation

If you find this repository, the code, or the methods useful for your academic work, please cite our paper:
```
@article{liu2024non,
  title={Non-binary evaluation of next-basket food recommendation},
  author={Liu, Yue and Achananuparp, Palakorn and Lim, Ee-Peng},
  journal={User Modeling and User-Adapted Interaction},
  volume={34},
  number={1},
  pages={183--227},
  year={2024},
  publisher={Springer}
}
```
