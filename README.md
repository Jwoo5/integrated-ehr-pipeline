# Integrated-EHR-Pipeline
- Pre-processing code refining project in [UniHPF](https://arxiv.org/abs/2207.09858)

## Install Requirements
- NOTE: This repository requires `python>=3.9` and `Java>=8`
```
pip install numpy pandas tqdm treelib transformers pyspark
```
## How to Use
```
main.py --ehr {eicu, mimiciii, mimiciv}
```
- It automatically download the corresponding dataset from physionet, but requires appropriate certification.
- You can also use the downloaded dataset with `--data {data path}` option
- You can check sample implementation of pytorch `dataset` on `sample_dataset.py`


- Run Command For Final Command
```
python main.py --ehr mimiciv --data /home/data_storage/MIMIC-IV-2.0/ --obs_size 48 --pred_size 48 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest /home/edlab/junukim/SNUB_UniHPF/ed/48h/ --num_threads 32 --readmission --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024" --use_ed

python main.py --ehr mimiciv --data /home/data_storage/MIMIC-IV-2.0/ --obs_size 48 --pred_size 24 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest /home/edlab/junukim/SNUB_UniHPF/ed/24h/ --num_threads 32 --readmission --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024" --use_ed

python main.py --ehr eicu --data /home/data_storage/eicu-2.0/ --obs_size 48 --pred_size 48 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest /home/edlab/junukim/SNUB_UniHPF/ed/48h/ --num_threads 32 --readmission --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024"

python main.py --ehr eicu --data /home/data_storage/eicu-2.0/ --obs_size 48 --pred_size 24 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest /home/edlab/junukim/SNUB_UniHPF/ed/24h/ --num_threads 32 --readmission --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024"


python main.py --ehr eicu --data /nfs_data_storage/AmsterdamUMCdb-v1.0.2/ --obs_size 48 --pred_size 48 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest /home/edlab/junukim/SNUB_UniHPF/ed/48h/ --num_threads 32 --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024"

python main.py --ehr eicu --data /nfs_data_storage/AmsterdamUMCdb-v1.0.2/ --obs_size 48 --pred_size 24 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest /home/edlab/junukim/SNUB_UniHPF/ed/24h/ --num_threads 32 --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024"

python main.py --ehr hirid --data /nfs_data_storage/hirid/1.1.1/ --obs_size 48 --pred_size 48 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest /home/edlab/junukim/SNUB_UniHPF/ed/48h/ --num_threads 32 --diagnosis --min_event_size 0 --seed 2020,2021,2022,2023,2024 

python main.py --ehr hirid --data /nfs_data_storage/hirid/1.1.1/ --obs_size 48 --pred_size 24 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest /home/edlab/junukim/SNUB_UniHPF/ed/24h/ --num_threads 32 --diagnosis --min_event_size 0 --seed 2020,2021,2022,2023,2024
```