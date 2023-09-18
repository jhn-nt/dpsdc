SELECT 
  icu.stay_id, 
  icu.gender,
  icu.admission_age,
  icu.hospital_expire_flag, 
  icu.los_icu,
  icustays.first_careunit,
  admissions.race,
  admissions.insurance,
  admissions.language,
  admissions.admission_location,
  patients.anchor_year_group,
  ventilation.starttime,
  DATETIME_DIFF(bg.bg_time, ventilation.starttime,MINUTE) AS time_to_abg__minutes, 
  DATETIME_DIFF(ventilation.endtime,ventilation.starttime,MINUTE)/60 AS duration__hours
FROM `physionet-data.mimiciv_derived.icustay_detail`icu 
INNER JOIN (
  SELECT DISTINCT 
    FIRST_VALUE(stay_id) OVER(PARTITION BY stay_id, ventilation_status ORDER BY starttime) AS stay_id,
    FIRST_VALUE(starttime) OVER(PARTITION BY stay_id, ventilation_status ORDER BY starttime) AS starttime,
    FIRST_VALUE(endtime) OVER(PARTITION BY stay_id, ventilation_status ORDER BY starttime) AS endtime,
    FIRST_VALUE(ventilation_status) OVER(PARTITION BY stay_id, ventilation_status ORDER BY starttime) AS ventilation_status
  FROM `physionet-data.mimiciv_derived.ventilation`
  WHERE ventilation_status='InvasiveVent') ventilation ON icu.stay_id=ventilation.stay_id AND DATETIME_DIFF(ventilation.starttime,icu.admittime,MINUTE)<360
INNER JOIN `physionet-data.mimiciv_hosp.admissions` admissions ON admissions.hadm_id=icu.hadm_id
INNER JOIN `physionet-data.mimiciv_icu.icustays` icustays ON icustays.stay_id=icu.stay_id
INNER JOIN `physionet-data.mimiciv_hosp.patients` patients ON patients.subject_id=admissions.subject_id
LEFT JOIN (
  SELECT hadm_id, MIN(charttime) AS bg_time
  FROM `physionet-data.mimiciv_derived.bg`
  WHERE specimen IN ('ART.','MIX.')
  GROUP BY hadm_id
  ) bg ON bg.hadm_id=icu.hadm_id AND bg.bg_time>=ventilation.starttime AND bg.bg_time<ventilation.endtime 
