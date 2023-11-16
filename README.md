# Integrated-EHR-Pipeline
- Pre-processing code refining project in [UniHPF](https://arxiv.org/abs/2207.09858)
- Modified for CXR_Pred

## NOTE: Delta from SNUB
- Migration to MIMIC-IV 2.2
- No DPE/Type
- Remove eICU support
- Support custom split


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
python main.py --ehr mimiciv --data /nfs_data_storage/mimic-iv-2.2 --obs_size 48 --pred_size 48 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest /nfs_edlab/junukim/CXR_Pred/48h/ --num_threads 32 --readmission --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024" --custom_split_path /nfs_edlab/dekyung/future_x-ray/data/datasplit_subject.json
```