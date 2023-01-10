# #whole text
# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr mimiciii \
# --first_icu \
# --data /home/data_storage/mimiciii-1.4/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --emb_type textbase --feature whole &&

# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr eicu \
# --first_icu \
# --data /home/data_storage/eicu-2.0/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --emb_type textbase --feature whole &&

# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr mimiciv \
# --first_icu \
# --data /home/data_storage/MIMIC-IV-2.0/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --emb_type textbase --feature whole &&

#whole code
# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr mimiciii \
# --first_icu \
# --data /home/data_storage/mimiciii-1.4/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --cache --emb_type codebase --feature whole &&

# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr eicu \
# --first_icu \
# --data /home/data_storage/eicu-2.0/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --emb_type codebase --feature whole &&

# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr mimiciv \
# --first_icu \
# --data /home/data_storage/MIMIC-IV-2.0/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --emb_type codebase --feature whole &&


# #select text
# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr mimiciii \
# --first_icu \
# --data /home/data_storage/mimiciii-1.4/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --emb_type textbase --feature "select" &&

# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr eicu \
# --first_icu \
# --data /home/data_storage/eicu-2.0/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --emb_type textbase --feature "select" &&

# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr mimiciv \
# --first_icu \
# --data /home/data_storage/MIMIC-IV-2.0/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --emb_type textbase --feature "select" &&

#select code
# python3 main.py \
# --dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
# --ehr mimiciii \
# --first_icu \
# --data /home/data_storage/mimiciii-1.4/ \
# --readmission --mortality --los_3day --los_7day --long_term_mortality \
# --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
#  --emb_type codebase --feature "select" &&

python3 main.py \
--dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
--ehr eicu \
--first_icu \
--data /home/data_storage/eicu-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
 --emb_type codebase --feature "select" &&

python3 main.py \
--dest /nfs_edlab/ghhur/UniHPF/input_test2/ \
--ehr mimiciv \
--first_icu \
--data /home/data_storage/MIMIC-IV-2.0/ \
--readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
 --emb_type codebase --feature "select" 
