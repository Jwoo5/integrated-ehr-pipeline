import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from ehrs import register_ehr, EHR
from utils import utils

logger = logging.getLogger(__name__)

@register_ehr('mimiciii')
class MIMICIII(EHR):
    def __init__(self, data, ccs, cfg):
        super().__init__()
        self.cfg = cfg

        self.dir_path = data
        self.ccs_path = ccs
        if data is None or ccs is None:
            if not os.path.exists('~/.cache'):
                os.mkdir('~/.cache')
            self.dir_path = os.path.abspath('~/.cache/mimiciii')
            self.ccs_path = os.path.abspath('~/.cache/mimiciii/ccs_multi_dx_tool_2015.csv')

        if not os.path.exists(self.dir_path):
            logger.info(
                'Data is not found so try to download from the internet.'
                'It requires ~7GB drive spaces.'
                'Note that this is a restricted-access resource.'
                'Please log in with a credentialed user.'
            )

            username = input('Email or Username: ')
            utils.runcmd(
                'wget -r -N -c -np --user '
                + username
                + ' --ask-password https://physionet.org/files/mimiciii/1.4/ -P ~/.cache',
                verbose=True
            )

            os.rename(
                '~/.cache/physionet.org/files/mimiciii/1.4',
                self.dir_path
            )
        if not os.path.exists(self.ccs_path):
            logger.info(
                '`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet.'
            )
            utils.runcmd(
                'wget https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip -P ~/.cache',
                verbose=True
            )
            import zipfile
            with zipfile.ZipFile(os.path.join(self.ccs_path, 'Multi_Level_CCS_2015.zip'), 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.ccs_path))

        self.icustays = 'ICUSTAYS.csv.gz'
        self.patients = 'PATIENTS.csv.gz'
        self.admissions = 'ADMISSIONS.csv.gz'
        self.diagnoses = 'DIAGNOSES_ICD.csv.gz'

        #XXX more features? user choice?
        self.features = [
            {
                'fname': 'LABEVENTS.csv.gz',
                'type': 'lab',
                'timestamp': 'CHARTTIME',
            },
            {
                'fname': 'PRESCRIPTIONS.csv.gz',
                'type': 'med',
                'timestamp': 'STARTDATE'
            },
            {
                'fname': 'INPUTEVENTS_MV.csv.gz',
                'type': 'inf',
                'timestamp': 'STARTTIME'
            },
            {
                'fname': 'INPUTEVENTS_CV.csv.gz',
                'type': 'inf',
                'timestamp': 'CHARTTIME'
            }
        ]

        self.max_event_size = cfg.max_event_size if cfg.max_event_size is not None else sys.maxsize
        self.min_event_size = cfg.min_event_size if cfg.min_event_size is not None else 0
        assert self.min_event_size <= self.max_event_size, (self.min_event_size, self.max_event_size)

        self.max_age = cfg.max_age if cfg.max_age is not None else sys.maxsize
        self.min_age = cfg.min_age if cfg.min_age is not None else 0
        assert self.min_age <= self.max_age, (self.min_age, self.max_age)

        self.obs_size = cfg.obs_size
        self.gap_size = cfg.gap_size
        self.pred_size = cfg.pred_size

    def build_cohort(self):
        patients = pd.read_csv(self.patients)
        icustays = pd.read_csv(self.icustays)
        admissions = pd.read_csv(self.admissions)

        icustays = icustays[icustays['FIRST_CAREUNIT'] == icustays['LAST_CAREUNIT']]
        icustays = icustays[icustays['LOS'] >= (self.obs_size + self.gap_size) / 24]
        icustays = icustays.drop(columns=['ROW_ID'])
        icustays['INTIME'] = pd.to_datetime(icustays['INTIME'], infer_datetime_format=True)
        icustays['OUTTIME'] = pd.to_datetime(icustays['OUTTIME'], infer_datetime_format=True)

        patients['DOB'] = pd.to_datetime(patients['DOB'], infer_datetime_format=True)
        patients = patients.drop(columns=['ROW_ID'])

        patients_with_icustays = patients[patients['SUBJECT_ID'].isin(icustays['SUBJECT_ID'])]
        patients_with_icustays = icustays.merge(patients_with_icustays, on='SUBJECT_ID', how='left')

        def calculate_age(birth: datetime, now: datetime):
            age = now.year - birth.year
            if now.month < birth.month:
                age -= 1
            elif (now.month == birth.month) and (now.day < birth.day):
                age -= 1
            
            return age

        patients_with_icustays['AGE'] = patients_with_icustays.apply(
            lambda x: calculate_age(x['DOB'],x['INTIME']), axis=1
        )
        patients_with_icustays = patients_with_icustays[
            (self.min_age <= patients_with_icustays['AGE'])
            & (patients_with_icustays['AGE'] <= self.max_age)
        ]

        # first icu
        is_readmitted = patients_with_icustays.groupby('HADM_ID')['ICUSTAY_ID'].count()
        is_readmitted = (is_readmitted > 1).astype(int).to_frame().rename(columns={'ICUSTAY_ID': 'readmission'})

        cohort = patients_with_icustays.loc[
            patients_with_icustays.groupby('HADM_ID')['INTIME'].idxmin()
        ]
        cohort = pd.merge(
            cohort.reset_index(drop=True),
            admissions[['HADM_ID', 'DISCHARGE_LOCATION', 'DEATHTIME', 'DISCHTIME']],
            how='left',
            on='HADM_ID'
        )
        cohort = cohort.join(is_readmitted, on='HADM_ID')
        cohort['DEATHTIME'] = pd.to_datetime(cohort['DEATHTIME'], infer_datetime_format=True)
        # XXX DISCHTIME --> HOSPITAL DISCHARGE TIME
        cohort['DISCHTIME'] = pd.to_datetime(cohort['DISCHTIME'], infer_datetime_format=True)

        self.cohort = cohort
        logger.info(
            'cohort has been built successfully. Loaded {} cohorts.'.format(len(self.cohort))
        )

        return cohort
    
    def prepare_tasks(self):
        #TODO check self.cohort
        breakpoint()
        # readmission prediction
        labeled_cohort = self.cohort[['HADM_ID', 'ICUSTAY_ID', 'readmission']]

        # los prediction
        labeled_cohort['los_3day'] = (self.cohort['LOS'] > 3).astype(int)
        labeled_cohort['los_7day'] = (self.cohort['LOS'] > 7).astype(int)

        # mortality prediction
        dead_patients = self.cohort[~self.cohort['DEATHTIME'].isna()]
        #TODO check if the same with self.cohort[~pd.isnull(self.cohort.DEATHTIME)]
        #TODO .copy()?
        breakpoint()
        is_dead = (
            (
                (
                    dead_patients['INTIME']
                    + pd.Timedelta(self.obs_size, unit='h')
                    + pd.Timedelta(self.gap_size, unit='h')
                ) <= dead_patients['DEATHTIME']
            )
            & (
                dead_patients['DEATHTIME'] <= (
                    dead_patients['INTIME']
                    + pd.Timedelta(self.obs_size, unit='h')
                    + pd.Timedelta(self.pred_size, unit='h')
                )
            )
        ).astype(int)
        dead_patients['mortality'] = np.array(is_dead)

        is_dead_in_icu = (
            dead_patients['DEATHTIME'] > dead_patients['INTIME']
        ) & (dead_patients['DEATHTIME'] <= dead_patients['OUTTIME'])
        dead_patients['in_icu_mortality'] = np.array(is_dead_in_icu.astype(int))

        labeled_cohort = pd.merge(
            labeled_cohort.reset_index(drop=True),
            dead_patients[['ICUSTAY_ID', 'mortality', 'in_icu_mortality']],
            on='ICUSTAY_ID',
            how='left'
        ).reset_index(drop=True)
        labeled_cohort['mortality'] = labeled_cohort['mortality'].fillna(0).astype(int)
        labeled_cohort['in_icu_mortality'] = labeled_cohort['in_icu_mortality'].fillna(0).astype(int)
        
        # final acuity prediction
        diagnoses = pd.read_csv(self.diagnoses)
        diagnoses_with_cohort = diagnoses[diagnoses['HADM_ID'].isin(labeled_cohort['HADM_ID'])]
        diagnoses_with_cohort = diagnoses_with_cohort.groupby('HADM_ID')['ICD9_CODE'].apply(list).to_frame()
        diagnoses_with_cohort = self.cohort.join(diagnoses_with_cohort, on='HADM_ID')

        diagnoses_with_cohort.loc[
            (diagnoses_with_cohort['DISCHARGE_LOCATION'] == 'DEAD/EXPIRED')
            & (diagnoses_with_cohort['in_icu_mortality'] == 0), 'in_hospital_mortality'
        ] = 1
        #TODO check
        breakpoint()
        diagnoses_with_cohort['in_hospital_mortality'] = (
            diagnoses_with_cohort['in_hospital_mortality'].fillna(0).astype(int)
        )
        diagnoses_with_cohort[
            diagnoses_with_cohort['in_icu_mortality'], 'DISCHARGE_LOCATION'
        ] = 'IN_ICU_MORTALITY'
        diagnoses_with_cohort[
            diagnoses_with_cohort['in_hospital_morality'], 'DISCHARGE_LOCATION'
        ] = 'IN_HOSPITAL_MORTALITY'

        diagnoses_with_cohort['final_acuity'] = (
            diagnoses_with_cohort['DISCHARGE_LOCATION'].astype('category').cat.codes
        )

        # imminent discharge prediction
        is_discharged = (
            (
                (
                    diagnoses_with_cohort.INTIME
                    + pd.Timedelta(self.obs_size, unit='h')
                    + pd.Timedelta(self.gap_size, unit='h')
                ) <= diagnoses_with_cohort['DISCHTIME']
            ) & (
                diagnoses_with_cohort['DISCHTIME'] <= (
                    diagnoses_with_cohort.INTIME
                    + pd.Timedelta(self.obs_size, unit='h')
                    + pd.Timedelta(self.pred_size, unit='h')
                )
            )
        ).astype(int)
        diagnoses_with_cohort.loc[
            is_discharged, 'imminent_discharge'
        ] = diagnoses_with_cohort[is_discharged]['DISCHARGE_LOCATION']
        diagnoses_with_cohort.loc[
            ~is_discharged, 'imminent_discharge'
        ] = 'No Discharge'
        diagnoses_with_cohort.loc[
            (diagnoses_with_cohort['imminent_discharge'] == 'IN_HOSPITAL_MORTALITY')
            | (diagnoses_with_cohort['imminent_discharge'] == 'IN_ICU_MORTALITY'),
            'imminent_discharge'
        ] = 'No Discharge'

        diagnoses_with_cohort['imminent_discharge'] = (
            diagnoses_with_cohort['imminent_discharge'].astype('category').cat.codes
        )

        labeled_cohort = diagnoses_with_cohort.drop(
            columns=['in_hospital_mortality', 'in_icu_mortality', 'DISCHARGE_LOCATION']
        )

        # diagnosis prediction
        ccs_dx = pd.read_csv(self.ccs_path)
        ccs_dx["'ICD-9-CM CODE\'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.replace(' ', '')
        ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
        lvl1 = {x: y for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()}

        dx1_list = []
        for dxs in labeled_cohort['ICD9_CODE']:
            one_list = []
            for dx in dxs:
                if dx not in lvl1:
                    continue
                dx1 = lvl1[dx]
                one_list.append(dx1)
            dx1_list.append(list(set(one_list)))

        labeled_cohort['diagnosis'] = pd.Series(dx1_list)
        labeled_cohort = labeled_cohort[labeled_cohort['diagnosis'] != float].reset_index(drop=True)

        self.labeled_cohort = labeled_cohort
        logger.info(
            'Done preparing tasks given the cohort sets'
        )

        return labeled_cohort

    def encode(self):
        encoded_cohort = self.labeled_cohort.rename(columns={'HADM_ID': 'ID'}, inplace=False)
        #TODO resume here
    
    def run_pipeline(self):
        ...