SELECT icu.stay_id,charlson.*
FROM `physionet-data.mimiciv_derived.charlson` charlson
INNER JOIN `physionet-data.mimiciv_derived.icustay_detail` icu ON icu.hadm_id=charlson.hadm_id 
INNER JOIN (
  SELECT DISTINCT 
    FIRST_VALUE(stay_id) OVER(PARTITION BY stay_id, ventilation_status ORDER BY starttime) AS stay_id,
    FIRST_VALUE(starttime) OVER(PARTITION BY stay_id, ventilation_status ORDER BY starttime) AS starttime,
    FIRST_VALUE(endtime) OVER(PARTITION BY stay_id, ventilation_status ORDER BY starttime) AS endtime,
    FIRST_VALUE(ventilation_status) OVER(PARTITION BY stay_id, ventilation_status ORDER BY starttime) AS ventilation_status
  FROM `physionet-data.mimiciv_derived.ventilation`
  WHERE ventilation_status='InvasiveVent') ventilation ON icu.stay_id=ventilation.stay_id AND DATETIME_DIFF(ventilation.starttime,icu.admittime,MINUTE)<360