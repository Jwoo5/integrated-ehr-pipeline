dest=$1
data=$2

echo "dest" $dest
echo "data" $data

#"whole" text
python3 main.py \
--dest $dest \
--ehr mimiciii \
--first_icu \
--data $data/mimiciii-1.4/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
--cache --emb_type textbase --feature "whole" &&

python3 main.py \
--dest $dest \
--ehr eicu \
--first_icu \
--data $data/eicu-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
--cache --emb_type textbase --feature "whole" &&

python3 main.py \
--dest $dest \
--ehr mimiciv \
--first_icu \
--data $data/MIMIC-IV-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
--cache --emb_type textbase --feature "whole" &&


#select text
python3 main.py \
--dest $dest \
--ehr mimiciii \
--first_icu \
--data $data/mimiciii-1.4/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
--cache --emb_type textbase --feature "select" &&

python3 main.py \
--dest $dest \
--ehr eicu \
--first_icu \
--data $data/eicu-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
--cache --emb_type textbase --feature "select" &&

python3 main.py \
--dest $dest \
--ehr mimiciv \
--first_icu \
--data $data/MIMIC-IV-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
--cache --emb_type textbase --feature "select" 