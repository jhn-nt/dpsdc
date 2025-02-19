# Care Phenotypes In Critical Care
LL. Weishaupt, T. Wang,  J. Schamroth,  P. Morandini,  J. Matos, LM Hampton,  J. Gallifant,  A. Fiske,  N. Dundas,  K. David,  LA. Celi,  A. Carrel, J. Byers,  G. Angelotti

Code for the _Care Phenotypes In Critical Care_ article, available in [preprint](https://www.medrxiv.org/content/10.1101/2025.01.24.25320468v1).  



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

### Delving Deeper  
The above code will run the experiment for a cohort of ventilated patients using weight as a disparity axis and frequency of turnings as a proxy. To test it on a different group run:
```python
python3 -m dpsdc -p <your project-id> -i /path/to/info.json -dp /path/to/proxy.sql -c /path/to/criteria.sql 
```
where:
1. `info.json` is used to charachterize the proxy variable, specifically, it requires 4 keys:
    -   `proxy_name`: The name of the proxy (Turnings).     
    -   `disparities_axis_name`: the disparity axis under which to measure the proxy (Weight).  
    -   `disparities_axis_uom`: The unit of measurements of the disparity axis (Kg).  
    -   `protocol__hours`: Exact, or when not possible, average frequency of proxy that patients should receive by protocol (for turnings, patients are expected to be turned every 2 hours).
See the example below:
```json
{
    "proxy_name":"Turnings",
    "disparities_axis_name":"Weight",
    "disparities_axis_uom":"Kg(s)",
    "protocol__hours":2
}
``` 
2. `criteria.sql` is a BigQuery query that when run, returns a list of patients ids which respect inclusion and exclusion criteria.
In the case of frequency of turnings:
```sql
SELECT DISTINCT 
    FIRST_VALUE(ventilation.stay_id) OVER(PARTITION BY ventilation.stay_id, ventilation.ventilation_status ORDER BY ventilation.starttime) AS stay_id,
FROM `physionet-data.mimiciv_derived.ventilation` ventilation
LEFT JOIN `physionet-data.mimiciv_derived.first_day_weight` weight ON weight.stay_id=ventilation.stay_id
WHERE ventilation.ventilation_status='InvasiveVent'
    AND weight.weight IS NOT NULL
    AND weight.weight>10
    AND weight.weight<250
    
```
For the turning example, we want only the first ventialtion instance of all patients whose weight is nor missing nor an outlier, defined as below 10Kgs or above 250Kgs.   
3. `proxy.sql` is a BigQuery query that when run, returns __daily__ proxy values we wish to invetigate __together with its patient id__. The patient id is necessary to crossreference the table with criteria defined above.  
Returning to the turnings example:
 ```sql
 WITH intubation AS (
  SELECT DISTINCT
    stay_id,
    FIRST_VALUE(starttime) OVER(PARTITION BY stay_id ORDER BY starttime) AS intubation_time,
    FIRST_VALUE(endtime) OVER(PARTITION BY stay_id ORDER BY starttime) AS extubation_time,
    FIRST_VALUE(ventilation_status) OVER(PARTITION BY stay_id ORDER BY starttime) AS ventilation_status

  FROM `physionet-data.mimiciv_derived.ventilation` 

  WHERE ventilation_status IN ('InvasiveVent')
)
SELECT 
  stay_id,
  day,
  24/COUNT(day) AS average_item_interval, 
  COUNT(day) AS item_volume,
  COUNT(DISTINCT caregiver_id) AS n_caregivers
FROM (
  SELECT 
    chartevents.stay_id,
    chartevents.caregiver_id,
    EXTRACT(DATE FROM intubation.intubation_time) AS intubation_date,
    EXTRACT(DATE FROM intubation.extubation_time) AS extubation_date,
    EXTRACT(DATE FROM chartevents.charttime) AS item_date,
    DENSE_RANK() OVER(PARTITION BY chartevents.stay_id ORDER BY EXTRACT(DATE FROM chartevents.charttime)) AS day
  FROM `physionet-data.mimiciv_icu.chartevents` chartevents
  INNER JOIN intubation ON chartevents.stay_id = intubation.stay_id
  WHERE itemid=224082 AND chartevents.charttime BETWEEN intubation.intubation_time AND intubation.extubation_time
    AND EXTRACT(DATE FROM intubation.intubation_time)!=EXTRACT(DATE FROM chartevents.charttime)
    AND EXTRACT(DATE FROM intubation.extubation_time)!=EXTRACT(DATE FROM chartevents.charttime)
    )
GROUP BY stay_id, day
```
The above query returns a dataset in with the following form:
| stay_id  | day | average_item_interval |
| ---------| ---- |--------- |
| pid1 | 01.02 | 2.35  |
| pid1 | 02.02| 1.98  |
| pid123| 12.09 | 2.12 |
| pid123| 13.09 | 2.25 |
| pid123| 14.09 | 1.93 |
| pid123| 15.09 | 3.21 |

You can further append auxialiary features in the query to be adjusted upon.

### Acknowledgements

1. [Johnson, Alistair, et al. "Mimic-iv." PhysioNet. Available online at: https://physionet. org/content/mimiciv/1.0/(accessed August 23, 2021) (2020).](https://physionet.org/content/mimiciv/2.1/)

2. [Yarnell, Christopher J., Alistair Johnson, Tariq Dam, Annemijn Jonkman, Kuan Liu, Hannah Wunsch, Laurent Brochard, et al. 2023. “Do Thresholds for Invasive Ventilation in Hypoxemic Respiratory Failure Exist? A Cohort Study.” American Journal of Respiratory and Critical Care Medicine 207 (3): 271–82.](https://pubmed.ncbi.nlm.nih.gov/36150166/)

3. [Wong, An-Kwok Ian, et al. "Analysis of discrepancies between pulse oximetry and arterial oxygen saturation measurements by race and ethnicity and association with organ dysfunction and mortality." JAMA network open 4.11 (2021): e2131674-e2131674.](https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2785794)

4. [Magesh, Shruti, Daniel John, Wei Tse Li, Yuxiang Li, Aidan Mattingly-App, Sharad Jain, Eric Y. Chang, and Weg M. Ongkeko. 2021. “Disparities in COVID-19 Outcomes by Race, Ethnicity, and Socioeconomic Status: A Systematic-Review and Meta-Analysis.” JAMA Network Open 4 (11): e2134147.](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2785980)
