SELECT icu.stay_id,charlson.*
FROM `physionet-data.mimiciv_derived.charlson` charlson
INNER JOIN `physionet-data.mimiciv_derived.icustay_detail` icu ON icu.hadm_id=charlson.hadm_id 
INNER JOIN ({}) inclusion_criteria ON inclusion_criteria.stay_id=icu.stay_id