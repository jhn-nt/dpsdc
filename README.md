# Disparity Proxies and Social Determinants of Care Workshop
## 

Invasive Mechnical Ventilation (IMV) is a complex treatment administered to ICU patients' with insufficient respiratory function.  
After intubation, IMV patients are closely monitored to ensure adequate oxygenation and ventilation. Several tools are available to assess the efficacy of IMV, and guidelines offer clear indications on their usage.   
Nevertheless, a wealth of evidence from reasearch [2][3][4] highlights how disadvantaged groups may experience unfair disparities in the quality of medical care they receive.  
It is our responsability as investigators to analyze data and formulate robust hypotheses to identify these disparities transparently, collaboratively and reproducibly.  
The following workshop is based on the MIMIC-IV database [1].


In this repo you will find all the code necessary to generate the tables and analysis used during the workshop.



### Requirements
1. Access to MIMIC-IV. Learn more [here](https://mimic.mit.edu/docs/gettingstarted/).
2. Project ID of a Google Project, make sure to have the necessary IAM permissions to run queries on Big Query.  

__Important Note__: The google account enabled to access the MIMIC-IV must the be _same_ as the one associated with the Google Project.  

### Installation
1. run `pip install "git+https://github.com/jhn-nt/dpsdc.git"`

### User Guide
Generating the report in folder `<working-directory>/output/`:
```python
python3 -m dpsdc <your project-id> -c <path/to/cohort/query>  -p <path/to/proxy/query>
```

Generating the report into a user defined folder:
 ```python
python3 -m  dpsdc -d <path/to/folder> <your project-id>
```

You will be prompted to authorize pandas GBQ to access your Google Account for the scope of the query.

Data will be downloaded only the first time that the package is run then stored in a temporary file for future reuses.   
To remove all temporary data run:
 ```python
python3 -m  dpsdc.reset
```

You can also navigate the data on your own; here is a selected set of commands that can be used in jupyter or colab notebooks:
 ```python
from dpsdc.etl import begin_workshop
from dpsdc.etl import load_profiles, load_view

# Initialization, required to download and process the data
begin_workshop(PROJECT_ID)

# Requesting all the features available in the workshop in the PORFILES dictionary.
PROFILES=load_profiles() 

# PROFILES maps features based on their nature: categorical, continuous or ordinal.
CATEGORICAL_FEATURES=PROFILES["categorical"]
CONTINUOUS_FEATURES=PROFILES["continuous"]
ORDINAL_FEATURES=PROFILES["ordinal"]

# Loading a data view based on user defined features into a pandas dataframe.
dataset=load_view(["admission_age","SOFA","hospital_death"],PROJECT_ID) 
```

### Citation
For those who desire to use the codebase from this repo in future projects, you can cite this workshop via the following:
```bibtex
Angelotti, G. (2023). Disparity Proxies and Social Determinants of Care Workshop. (Version 0.1) [Computer software]. https://github.com/jhn-nt/dpsdc
```

### Acknowledgements

1. [Johnson, Alistair, et al. "Mimic-iv." PhysioNet. Available online at: https://physionet. org/content/mimiciv/1.0/(accessed August 23, 2021) (2020).](https://physionet.org/content/mimiciv/2.1/)

2. [Yarnell, Christopher J., Alistair Johnson, Tariq Dam, Annemijn Jonkman, Kuan Liu, Hannah Wunsch, Laurent Brochard, et al. 2023. “Do Thresholds for Invasive Ventilation in Hypoxemic Respiratory Failure Exist? A Cohort Study.” American Journal of Respiratory and Critical Care Medicine 207 (3): 271–82.](https://pubmed.ncbi.nlm.nih.gov/36150166/)

3. [Wong, An-Kwok Ian, et al. "Analysis of discrepancies between pulse oximetry and arterial oxygen saturation measurements by race and ethnicity and association with organ dysfunction and mortality." JAMA network open 4.11 (2021): e2131674-e2131674.](https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2785794)

4. [Magesh, Shruti, Daniel John, Wei Tse Li, Yuxiang Li, Aidan Mattingly-App, Sharad Jain, Eric Y. Chang, and Weg M. Ongkeko. 2021. “Disparities in COVID-19 Outcomes by Race, Ethnicity, and Socioeconomic Status: A Systematic-Review and Meta-Analysis.” JAMA Network Open 4 (11): e2134147.](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2785980)
