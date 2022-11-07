# Integrated-EHR-Pipeline
- Pre-processing code refining project in [UniHPF](https://arxiv.org/abs/2207.09858)

## Install Requirements
```
pip install numpy pandas sortedcontainers tqdm treelib transformers
```
## How to Use
```
main.py --ehr {eicu, mimiciii, mimiciv}
```
- It automatically download the corresponding dataset from physionet, but requires appropriate certification.
- You can also use the downloaded dataset with `--data {data path}` option
- You can check sample implementation of pytorch `dataset` on `sample_dataset.py`
