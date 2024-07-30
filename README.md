# Adherence to Frequency of Turnings Protocol in Mechanically Ventilated Patients in MIMIC-IV
###



### Requirements
1. Access to MIMIC-IV. Learn more [here](https://mimic.mit.edu/docs/gettingstarted/).
2. Project ID of a Google Project, make sure to have the necessary IAM permissions to run queries on Big Query.  

__Important Note__: The google account enabled to access the MIMIC-IV must the be _same_ as the one associated with the Google Project.  

### Installation
1. run `pip install "git+https://github.com/jhn-nt/dpsdc.git"`  
2. to test the installation, run `python3 -m unittest dpsdc.tests` 

### Run the experiment
1. dry run, completed in a couple of minutes: `python3 -m dpsdc -p <your project-id> --dry` 
2. to run  the whole experiment (ca 2 hours), use: `python3 -m dpsdc -p <your project-id>` 

The above code will run the experiment for a cohort of ventilated patients using weight as a disparity axis and frequency of turnings as a proxy. To test it on a different group run:
```python
python3 -m dpsdc -p <your project-id> -i /path/to/info.json -dp /path/to/proxy.sql -c /path/to/cohort.sql 
```
where:
1. `info.json` is json containing three keys
2. `cohort.sql` is a BigQuery query returning a list of patients ids of the cohort under investigation
3. `proxy.sql` A proxy quantity that one wants to investigate. Each rows shoould contain also a patient id





### Acknowledgements

1. [Johnson, Alistair, et al. "Mimic-iv." PhysioNet. Available online at: https://physionet. org/content/mimiciv/1.0/(accessed August 23, 2021) (2020).](https://physionet.org/content/mimiciv/2.1/)

2. [Yarnell, Christopher J., Alistair Johnson, Tariq Dam, Annemijn Jonkman, Kuan Liu, Hannah Wunsch, Laurent Brochard, et al. 2023. “Do Thresholds for Invasive Ventilation in Hypoxemic Respiratory Failure Exist? A Cohort Study.” American Journal of Respiratory and Critical Care Medicine 207 (3): 271–82.](https://pubmed.ncbi.nlm.nih.gov/36150166/)

3. [Wong, An-Kwok Ian, et al. "Analysis of discrepancies between pulse oximetry and arterial oxygen saturation measurements by race and ethnicity and association with organ dysfunction and mortality." JAMA network open 4.11 (2021): e2131674-e2131674.](https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2785794)

4. [Magesh, Shruti, Daniel John, Wei Tse Li, Yuxiang Li, Aidan Mattingly-App, Sharad Jain, Eric Y. Chang, and Weg M. Ongkeko. 2021. “Disparities in COVID-19 Outcomes by Race, Ethnicity, and Socioeconomic Status: A Systematic-Review and Meta-Analysis.” JAMA Network Open 4 (11): e2134147.](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2785980)
