# Gravitational Waves (A5) Coursework Repository

This repository contains the report and supporting code for the Gravitational Waves (A5) coursework.


---

## Installation Instructions

To run the notebooks, please follow these steps:

### 1. Clone the Repository

Clone the repository from the remote repository (GitLab) to your local machine.

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a5_coursework/jlt67.git
cd jlt67
```

### 2. Create and Activate the Conda Environment
We recommend using the provided environment_GW.yml file to create a clean Conda environmentto avoid dependency conflicts.
Use a clean virtual environment to avoid dependency conflicts.
```bash
conda env create -f environment_GW.yml
conda activate GW_Coursework
```


### 4. Set Up a Jupyter Notebook Kernel
To ensure the conda environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Python (GW/A5 Coursework)"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel **Python (GW/A5 Coursework)** to run the code.


## For Assessment

### Report
Please find the projects report under `Report` directory

### Declaration of Use of Autogeneration Tools
This report made use of Large Language Models (LLMs) to assist in the development of the project.
These tools have been employed for:
- Formatting plots to enhance presentation quality.
- Performing iterative changes to already defined code.
- Debugging code and identifying issues in implementation.
- Helping with Latex formatting for the report.
- Identifying grammar and punctuation inconsistencies within the report.