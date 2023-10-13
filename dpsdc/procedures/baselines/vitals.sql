SELECT vitalsign.*
FROM `physionet-data.mimiciv_derived.first_day_vitalsign` vitalsign
INNER JOIN `physionet-data.mimiciv_derived.icustay_detail` icu ON icu.stay_id=vitalsign.stay_id 
INNER JOIN ({}) inclusion_criteria ON inclusion_criteria.stay_id=icu.stay_id