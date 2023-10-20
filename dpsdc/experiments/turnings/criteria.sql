SELECT DISTINCT 
    FIRST_VALUE(ventilation.stay_id) OVER(PARTITION BY ventilation.stay_id, ventilation.ventilation_status ORDER BY ventilation.starttime) AS stay_id,
    FIRST_VALUE(ventilation.starttime) OVER(PARTITION BY ventilation.stay_id, ventilation.ventilation_status ORDER BY ventilation.starttime) AS starttime,
    FIRST_VALUE(ventilation.endtime) OVER(PARTITION BY ventilation.stay_id, ventilation.ventilation_status ORDER BY ventilation.starttime) AS endtime,
    FIRST_VALUE(ventilation.ventilation_status) OVER(PARTITION BY ventilation.stay_id, ventilation.ventilation_status ORDER BY ventilation.starttime) AS ventilation_status
FROM `physionet-data.mimiciv_derived.ventilation` ventilation
LEFT JOIN `physionet-data.mimiciv_derived.first_day_weight` weight ON weight.stay_id=ventilation.stay_id
WHERE ventilation.ventilation_status='InvasiveVent'
    AND weight.weight IS NOT NULL
    