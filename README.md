# Disparity Proxies and Social Determinants of Care 
## Workshop

This repo contains all the code necessary to reproduce the reports and data tables used in the .


### Requirements
1. Access to MIMIC-IV. Learn more [here](https://mimic.mit.edu/docs/gettingstarted/).
2. Project ID of a Google Project, make sure to have the necessary IAM permissions to run queries on Big Query.  

The google account enabled to access the MIMIC-IV must the be same as the one associated with the Google Project. 

### Installation
1. run `pip install "git+https://github.com/jhn-nt/dpsdc.git"`

### User Guide
Generating the report in folder `<working-directory>/output/`:
```python
python3 dpsdc -p <your project-id>
```

Generating the report into a user defined folder:
 ```python
python3 dpsdc -d <path/to/folder> -p <your project-id> 
```