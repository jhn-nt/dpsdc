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
  height.height,
  weight.weight,
FROM `physionet-data.mimiciv_derived.icustay_detail`icu 
INNER JOIN `physionet-data.mimiciv_hosp.admissions` admissions ON admissions.hadm_id=icu.hadm_id
INNER JOIN `physionet-data.mimiciv_icu.icustays` icustays ON icustays.stay_id=icu.stay_id
INNER JOIN `physionet-data.mimiciv_hosp.patients` patients ON patients.subject_id=admissions.subject_id
LEFT JOIN `physionet-data.mimiciv_derived.first_day_height` height ON height.stay_id=icu.stay_id
LEFT JOIN `physionet-data.mimiciv_derived.first_day_weight` weight ON weight.stay_id=icu.stay_id
INNER JOIN ({}) inclusion_criteria ON inclusion_criteria.stay_id=icu.stay_id


