# Integrated-EHR-Pipeline
- Pre-processing code refining project in [UniHPF](https://arxiv.org/abs/2207.09858)

## Install Requirements
- NOTE: This repository requires `python>=3.9` and `Java>=8`
```
sudo apt update && sudo apt install openjdk-8-jdk
pip install numpy pandas tqdm transformers pyspark scikit-learn
```
## How to Use
```
main.py --ehr {eicu, mimiciii, mimiciv}
```
- It automatically download the corresponding dataset from physionet, but requires appropriate certification.
- You can also use the downloaded dataset with `--data {data path}` option
- You can check sample implementation of pytorch `dataset` on `sample_dataset.py`


- Run Command
```bash
python main.py --ehr mimiciv --data /nfs_data_storage/mimic-iv-2.2/ --obs_size 0 --pred_size 24 --max_patient_token_len 2147483647 --max_event_size 2147483647 --dest /nfs_edlab/junukim/LLM_Pred_data/in_icu_mort/ --num_threads 32 --min_event_size 5 --seed "2020, 2021, 2022, 2023, 2024"
```

```bash
python main.py --ehr mimiciv --data /nfs_data_storage/eicu/ --obs_size 0 --pred_size 24 --max_patient_token_len 2147483647 --max_event_size 2147483647 --dest /nfs_edlab/junukim/LLM_Pred_data/in_icu_mort/ --num_threads 32 --min_event_size 5 --seed "2020, 2021, 2022, 2023, 2024"
```