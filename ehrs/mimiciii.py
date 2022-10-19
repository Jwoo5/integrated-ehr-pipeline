import os
import sys
import subprocess
import shutil
import logging

from datetime import datetime
import numpy as np
import pandas as pd

from ehrs import register_ehr, EHR

logger = logging.getLogger(__name__)

@register_ehr('mimiciii')
class MIMICIII(EHR):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.data_dir = cfg.data
        self.ccs_path = cfg.ccs

        cache_dir = os.path.expanduser('~/.cache/ehr')

        if self.data_dir is None or not os.path.exists(self.data_dir):
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            self.data_dir = os.path.join(cache_dir, 'mimiciii')

            if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) == 30:
                logger.info(
                    'Loaded cached ehr data from {}.'.format(self.data_dir)
                )
            else:
                logger.info(
                    'Data is not found so try to download from the internet. '
                    'It requires ~7GB drive spaces. '
                    'Note that this is a restricted-access resource. '
                    'Please log in to physionet.org with a credentialed user.'
                )

                username = input('Email or Username: ')
                subprocess.run([
                    'wget', '-r', '-N', '-c', 'np', '--user', username,
                    '--ask-password', 'https://physionet.org/files/mimiciii/1.4/', '-P', cache_dir
                ])

                if len(os.listdir(os.path.join(cache_dir, 'physionet.org/files/mimiciii/1.4'))) != 30:
                    raise AssertionError(
                        'Access refused. Please log in with a credentialed user.'
                    )

                os.rename(
                    os.path.join(cache_dir, 'physionet.org/files/mimiciii/1.4'),
                    self.data_dir
                )
                shutil.rmtree(os.path.join(cache_dir, 'physionet.org'))
        if self.ccs_path is None or not os.path.exists(self.ccs_path):
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            self.ccs_path = os.path.join(cache_dir, 'ccs_multi_dx_tool_2015.csv')

            if os.path.exists(self.ccs_path):
                logger.info(
                    'Loaded cached ccs file from {}'.format(self.ccs_path)
                )
            else:
                logger.info(
                    '`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet.'
                )

                subprocess.run([
                    'wget', 'https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip',
                    '-P', cache_dir
                ])

                import zipfile
                with zipfile.ZipFile(os.path.join(cache_dir,'Multi_Level_CCS_2015.zip'), 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(cache_dir, 'tmp'))
                os.rename(
                    os.path.join(cache_dir, 'tmp', 'ccs_multi_dx_tool_2015.csv'),
                    self.ccs_path
                )
                os.remove(os.path.join(cache_dir, 'Multi_Level_CCS_2015.zip'))
                shutil.rmtree(os.path.join(cache_dir, 'tmp'))

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

        self.first_icu = cfg.first_icu

    def build_cohort(self):
        patients = pd.read_csv(os.path.join(self.data_dir,self.patients))
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustays))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admissions))

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

        # merge with admissions to get discharge information
        patients_with_icustays = pd.merge(
            patients_with_icustays.reset_index(drop=True),
            admissions[['HADM_ID', 'DISCHARGE_LOCATION', 'DEATHTIME', 'DISCHTIME']],
            how='left',
            on='HADM_ID'
        )

        # we define labels for the readmission task in this step
        # since it requires to observe each next icustays,
        # which would have been excluded in the final cohorts
        if self.first_icu:
            # check if each HADM_ID has multiple icustays
            is_readmitted = patients_with_icustays.groupby('HADM_ID')['ICUSTAY_ID'].count()
            is_readmitted = (is_readmitted > 1).astype(int).to_frame().rename(columns={'ICUSTAY_ID': 'readmission'})

            # take the first icustays for each HADM_ID
            patients_with_icustays = patients_with_icustays.loc[
                patients_with_icustays.groupby('HADM_ID')['INTIME'].idxmin()
            ]
            # assign an appropriate label for the readmission task
            patients_with_icustays = patients_with_icustays.join(is_readmitted, on='HADM_ID')
        else:
            patients_with_icustays['readmission'] = 1
            # the last icustay for each HADM_ID means that they have no icu readmission
            patients_with_icustays.loc[
                patients_with_icustays.groupby('HADM_ID')['INTIME'].idxmax(), 'readmission'
            ] = 0

        patients_with_icustays['DEATHTIME'] = pd.to_datetime(patients_with_icustays['DEATHTIME'], infer_datetime_format=True)
        # XXX DISCHTIME --> HOSPITAL DISCHARGE TIME
        patients_with_icustays['DISCHTIME'] = pd.to_datetime(patients_with_icustays['DISCHTIME'], infer_datetime_format=True)

        self.cohort = patients_with_icustays
        logger.info(
            'cohort has been built successfully. Loaded {} cohorts.'.format(len(self.cohort))
        )

        return patients_with_icustays
    
    def prepare_tasks(self):
        # readmission prediction
        labeled_cohort = self.cohort[['HADM_ID', 'ICUSTAY_ID', 'readmission']].copy()

        # los prediction
        labeled_cohort['los_3day'] = (self.cohort['LOS'] > 3).astype(int)
        labeled_cohort['los_7day'] = (self.cohort['LOS'] > 7).astype(int)

        # mortality prediction
        # filter out dead patients
        dead_patients = self.cohort[~self.cohort['DEATHTIME'].isna()]
        dead_patients = dead_patients[['ICUSTAY_ID', 'INTIME', 'OUTTIME', 'DEATHTIME']].copy()

        # if intime + obs_size + gap_size <= deathtime <= intime + obs_size + pred_size
        # it is assigned positive label on the mortality prediction
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

        # if icu intime < deathtime <= icu outtime
        # we also retain this case as in_icu_mortality for the imminent discharge task
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

        # join with self.cohort to get information needed for imminent discharge task
        labeled_cohort = labeled_cohort.join(
            self.cohort[
                ['ICUSTAY_ID', 'INTIME', 'DISCHTIME', 'DISCHARGE_LOCATION']
            ].set_index('ICUSTAY_ID'), on='ICUSTAY_ID'
        )

        # if an icustay is DEAD/EXPIRED, but not in_icu_mortality, then it is in_hospital_mortality
        labeled_cohort['in_hospital_mortality'] = 0
        labeled_cohort.loc[
            (labeled_cohort['DISCHARGE_LOCATION'] == 'DEAD/EXPIRED')
            & (labeled_cohort['in_icu_mortality'] == 0), 'in_hospital_mortality'
        ] = 1
        # define new class whose discharge location was 'DEAD/EXPIRED'
        labeled_cohort.loc[
            labeled_cohort['in_icu_mortality'] == 1, 'DISCHARGE_LOCATION'
        ] = 'IN_ICU_MORTALITY'
        labeled_cohort.loc[
            labeled_cohort['in_hospital_mortality'] == 1, 'DISCHARGE_LOCATION'
        ] = 'IN_HOSPITAL_MORTALITY'

        # define final acuity prediction task
        labeled_cohort['final_acuity'] = (
            labeled_cohort['DISCHARGE_LOCATION'].astype('category').cat.codes
        )

        # define imminent discharge prediction task
        is_discharged = (
            (
                (
                    labeled_cohort['INTIME']
                    + pd.Timedelta(self.obs_size, unit='h')
                    + pd.Timedelta(self.gap_size, unit='h')
                ) <= labeled_cohort['DISCHTIME']
            ) & (
                labeled_cohort['DISCHTIME'] <= (
                    labeled_cohort['INTIME']
                    + pd.Timedelta(self.obs_size, unit='h')
                    + pd.Timedelta(self.pred_size, unit='h')
                )
            )
        ).astype(bool)
        labeled_cohort.loc[
            is_discharged, 'imminent_discharge'
        ] = labeled_cohort[is_discharged]['DISCHARGE_LOCATION']
        labeled_cohort.loc[
            ~is_discharged, 'imminent_discharge'
        ] = 'No Discharge'
        labeled_cohort.loc[
            (labeled_cohort['imminent_discharge'] == 'IN_HOSPITAL_MORTALITY')
            | (labeled_cohort['imminent_discharge'] == 'IN_ICU_MORTALITY'),
            'imminent_discharge'
        ] = 'Death'

        labeled_cohort['imminent_discharge'] = (
            labeled_cohort['imminent_discharge'].astype('category').cat.codes
        )

        # drop unnecessary columns
        labeled_cohort = labeled_cohort.drop(
            columns=['in_icu_mortality', 'INTIME', 'DISCHTIME', 'DISCHARGE_LOCATION', 'in_hospital_mortality']
        )

        # define diagnosis prediction task
        diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnoses))

        diagnoses_with_cohort = diagnoses[diagnoses['HADM_ID'].isin(labeled_cohort['HADM_ID'])]
        diagnoses_with_cohort = diagnoses_with_cohort.groupby('HADM_ID')['ICD9_CODE'].apply(list).to_frame()
        labeled_cohort = labeled_cohort.join(diagnoses_with_cohort, on='HADM_ID')

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
        #XXX what does this line do?
        labeled_cohort = labeled_cohort[labeled_cohort['diagnosis'] != float].reset_index(drop=True)
        labeled_cohort = labeled_cohort.drop(columns=['ICD9_CODE'])

        self.labeled_cohort = labeled_cohort
        logger.info(
            'Done preparing tasks given the cohort sets'
        )

        return labeled_cohort

    def encode(self):
        encoded_cohort = self.labeled_cohort.rename(columns={'HADM_ID': 'ID'}, inplace=False)
        #TODO resume here
    
    # def run_pipeline(self):
    #     ...