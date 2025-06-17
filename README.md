# ccm-code

Project code and data for PIECE manuscript to get started on creating "toolboxes" for the lab.

1) To install the same Python packages and versions used for all analyses, you can recreate the same virtual environment by entering the following command in your terminal (within your cloned/downloaded project directory):

    `$ conda env create --name <nameyourenvironment> --file environment.yml`

2) These notebooks call on a package of custom-written functions found in the `src` directory. You may need to pip install this package. You can do so with the following command:

    `pip install -e .`


3) Once you've installed and conda activated the replicated environment, and pip installed the source code, you should be able to seamlessly run the Jupyter notebooks found in `experiment-1/scripts` to recreate the analyses:
    - `analyze-behavior.ipynb`
    - `model-fitting.ipynb`
    - `parameter-model-recovery.ipynb`