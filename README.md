# Ani-RNA

## Contents
- [py3.12_env.yaml](#py312_envyaml)
- [Example Datasets](#example-datasets)
- [OutSingle Tool](#outsingle-tool)
- [functions.py](#functionspy)
- [Installation and Usage](#installation-and-usage)
- [Reproducibility](#reproducibility)

### py3.12_env.yaml
This file exports the software environment.

### Example Datasets  
Example datasets for demonstration, located in the `example` folder.  

### OutSingle Tool  
An expression outlier detection tool used in the analysis pipeline, located in `outsingle` folder.  

### functions.py  
Essential functions for the analysis, called during the process.

### Installation and Usage
1. **Install Python Environment**:
   ```bash
   conda env create -f py3.12_env.yaml
   ```
2. **Run the Analysis**:
   - **OutSingle Analysis**:
     - Open `ae_outsingle.ipynb`.
     - Import the dataset from the `example` folder.
     - Run the notebook to obtain OutSingle results.
   - **Aberrant Gene Expression Network Analysis**:
     - Open `dataset_run.ipynb`.
     - Execute the aberrant gene expression network analysis.
     - The analysis will generate AE-network node and edge files.
     - Note: The analysis process will call functions from `functions.py`.

### Reproducibility
- To reproduce the enrichment analysis of AE-network node gene sets as described in our manuscript, run the `gsea.ipynb` notebook.

---

*For any questions or issues, please refer to the documentation or contact me.
