dest=$1
data=$2

echo "dest" $dest
echo "data" $data


#whole code
python3 main.py \
--dest $dest \
--ehr mimiciii \
--first_icu \
--data $2/mimiciii-1.4/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
 --emb_type codebase --feature "whole" &&

python3 main.py \
--dest $dest \
--ehr eicu \
--first_icu \
--data $2/eicu-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
 --emb_type codebase --feature "whole" &&

python3 main.py \
--dest $dest \
--ehr mimiciv \
--first_icu \
--data $2/MIMIC-IV-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
 --emb_type codebase --feature "whole" &&


#select code
python3 main.py \
--dest $dest \
--ehr mimiciii \
--first_icu \
--data $2/mimiciii-1.4/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
 --emb_type codebase --feature "select" &&

python3 main.py \
--dest $dest \
--ehr eicu \
--first_icu \
--data $2/eicu-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
 --emb_type codebase --feature "select" &&

python3 main.py \
--dest $dest \
--ehr mimiciv \
--first_icu \
--data $2/MIMIC-IV-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
 --emb_type codebase --feature "select" 
