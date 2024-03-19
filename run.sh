#!/bin/bash

# 인자 파싱
if [ $# -lt 1 ]; then
    echo "Usage: $0 --data <dataset_name> [--cache]"
    exit 1
fi

DATASET=""
CACHE_FLAG=""

for i in "$@"; do
case $i in
    --data)
    DATASET="$2"
    shift # past argument
    shift # past value
    ;;
    --cache)
    CACHE_FLAG="--cache"
    shift # past argument
    ;;
    *)
    # unknown option
    ;;
esac
done

# 데이터셋 이름이 제공되지 않은 경우 에러 메시지 출력 및 종료
if [ -z "$DATASET" ]; then
    echo "Error: Dataset name not provided."
    exit 1
fi

# 데이터셋 및 해당 설정들을 순회
declare -A datasets=(
    [mimiciv]="/home/data_storage/MIMIC-IV-2.0/"
    [eicu]="/home/data_storage/eicu-2.0/"
    [umcdb]="/nfs_data_storage/AmsterdamUMCdb-v1.0.2/"
    [hirid]="/nfs_data_storage/hirid/1.1.1/"
)

# 데이터셋 이름이 유효한지 확인
if [[ ! ${datasets[$DATASET]+_} ]]; then
    echo "Error: Dataset $DATASET is not valid. Choose from mimiciv, eicu, umcdb, hirid."
    exit 1
fi

# 로그 파일 이름 설정
LOG_FILE="execution_time_log_${DATASET}.txt"

# 실행 시간 로그 파일 초기화
echo "Execution Time Log for $DATASET" > $LOG_FILE

data_path=${datasets[$DATASET]}

# obs_size와 pred_size 설정
sizes=("48 48" "48 24")

for size in "${sizes[@]}"; do
    read obs_size pred_size <<< "$size"
    dest_path="/nfs_edlab/ghhur/projects/multi-lingual/datasets/ed/${pred_size}h/"
    common_args="--ehr $DATASET --data $data_path --obs_size $obs_size --pred_size $pred_size --max_patient_token_len 2147483647 --max_event_size 2147483647 ${CACHE_FLAG} --use_more_tables --dest $dest_path --num_threads 32 --diagnosis --min_event_size 0 --seed \"2020, 2021, 2022, 2023, 2024\""

    # mimiciv 또는 eicu일 경우 --readmission 추가
    # mimiciv 인 경우에만 --use_ed 추가
    if [[ "$DATASET" == "mimiciv" ]]; then
        common_args+=" --use_ed --readmission"
    elif [[ "$DATASET" == "eicu" ]]; then
        common_args+=" --readmission"
    fi

    # 실행 시간 측정 및 실행
    start_time=$(date +%s)
    eval python main.py $common_args
    end_time=$(date +%s)

    # 로그 기록
    echo "Dataset: $DATASET, obs_size: $obs_size, pred_size: $pred_size" >> $LOG_FILE
    echo "Execution Time: $((end_time - start_time)) seconds" >> $LOG_FILE
    echo "-----------------------------------" >> $LOG_FILE
done
