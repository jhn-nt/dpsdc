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
  WHERE itemid=226168 AND value != 'Swab' AND chartevents.charttime BETWEEN intubation.intubation_time AND intubation.extubation_time
    AND EXTRACT(DATE FROM intubation.intubation_time)!=EXTRACT(DATE FROM chartevents.charttime)
    AND EXTRACT(DATE FROM intubation.extubation_time)!=EXTRACT(DATE FROM chartevents.charttime))
GROUP BY stay_id, day