import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.impute import KNNImputer
from tqdm import tqdm

import gzip
import shutil
import os
from glob import glob
import re
from datetime import datetime as dt

# ================================================================================================
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ================================================================================================

# handle the gzip decompression beforehand
# 
# the reason is that when loading directly from a .csv.gzip file,
# dask does not seem to be able to consistently split the dataframe into chunks,
# so we lose the main reason why we use dask over pandas

hosp_dir = '/PATH/TO/physionet.org/files/mimiciv/3.0/hosp/'
icu_dir = '/PATH/TO/physionet.org/files/mimiciv/3.0/icu/'
data_dir = '/PATH/TO/data' # for intermediate data between each step and the final csv

hosp_files = []
icu_files = []

def ungzip_required(directory, required_files):

	gzip_files = glob(os.path.join(directory, '*.csv.gz'))
	filtered_files = [f for f in gzip_files if any(req in os.path.basename(f) for req in required_files)]

	for gzip_file in filtered_files:
		base_name = os.path.basename(gzip_file[:-3]) # removes '.gz'
		csv_file = os.path.join(directory, base_name)

		if os.path.exists(csv_file): continue

		with gzip.open(gzip_file, 'rb') as f_in:
			with open(csv_file, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)

# ================================================================================================
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ================================================================================================

icu_files.append('chartevents')

def step_one_charts():
	print(dt.now().time(), '1. chartevents')
	icu_charts = dd.read_csv(os.path.join(icu_dir, 'chartevents.csv'), dtype = {
		'subject_id': 'int64',
		'hadm_id': 'int64',
		'stay_id': 'Int64',
		'caregiver_id': 'Int64',
		'charttime': 'string',
		'storetime': 'string',
		'itemid': 'int64',
		'value': 'string',
		'valuenum': 'Float64',
		'valueuom': 'string',
		'warning': 'Int64'
	}, usecols = [
		'hadm_id',
		'charttime',
		'itemid',
		'valuenum',
	])
           
	charts_features = {
		227073: 'anion_gap',
		225310: 'art_bp_diastolic',
		225309: 'art_bp_systolic',
		220051: 'arterial_blood_pressure_diastolic',
		220050: 'arterial_blood_pressure_systolic',
		225624: 'bun',
		220074: 'central_venous_pressure',
		220602: 'chloride',
		220615: 'creatinine',
		224639: 'daily_weight',
		225637: 'atypical_lymphocytes',
		225638: 'bands',
		225639: 'basophils',
		225640: 'eosinophils',
		225641: 'lymphocytes',
		225642: 'monocytes',
		225643: 'neutrophils',
		220621: 'glucose',
		220624: 'hdl',
		220045: 'heart_rate',
		220545: 'hematocrit',
		220228: 'hemoglobin',
		227467: 'inr',
		223835: 'inspired_O2_fraction',
		227441: 'ldl_measured',
		220180: 'non_invasive_blood_pressure_diastolic',
		220179: 'non_invasive_blood_pressure_systolic',
		220277: 'O2_saturation_pulseoxymetry',
		220339: 'peep_set',
		227457: 'platelet_count',
		227442: 'potassium',
		227466: 'ptt',
		220210: 'respiratory_rate',
		220645: 'sodium',
		223761: 'temperature_fahrenheit',
		224684: 'tidal_volume',
		225108: 'tobacco_use',
		225693: 'triglyceride',
		227429: 'troponin-t',
		220546: 'white_blood_cell'
	}

	#
	print(dt.now().time(), 'filtering to relevant features and creating index...')
	charts = icu_charts[icu_charts['itemid'].isin(charts_features.keys())].compute()

	charts['chartday'] = charts['charttime'].astype('str').str.split(' ').apply(lambda x: x[0])
	charts['HADMID_DAY'] = charts['hadm_id'].astype('str') + '_' + charts['chartday']
	charts['FEATURES'] = charts['itemid'].apply(lambda x: charts_features[x])

	#
	print(dt.now().time(), 'creating daily aggregates...')
	# aggregate daily values to median, iqr, min, max, and concat those into a large pivoted table
	# - median and iqr were chosen because they are robust to outliers and non-normal distributions
	# - min and max were chosen to still inform the model of the range of values
	charts_pivoted_median = pd.pivot_table(charts, index='HADMID_DAY', columns='FEATURES', values='valuenum', aggfunc='median', fill_value=np.nan, dropna=False, sort=False)

	def iqr(x): return x.quantile(0.75) - x.quantile(0.25)
	charts_pivoted_iqr = pd.pivot_table(charts, index='HADMID_DAY', columns='FEATURES', values='valuenum', aggfunc=iqr, fill_value=0, dropna=False, sort=False)
	charts_pivoted_iqr.columns = [f"{col}_iqr" for col in charts_pivoted_iqr.columns]

	charts_pivoted_min = pd.pivot_table(charts, index='HADMID_DAY', columns='FEATURES', values='valuenum', aggfunc='min', fill_value=np.nan, dropna=False, sort=False)
	charts_pivoted_min.columns = [f"{col}_min" for col in charts_pivoted_min.columns]

	charts_pivoted_max = pd.pivot_table(charts, index='HADMID_DAY', columns='FEATURES', values='valuenum', aggfunc='max', fill_value=np.nan, dropna=False, sort=False)
	charts_pivoted_max.columns = [f"{col}_max" for col in charts_pivoted_max.columns]

	charts_concat = pd.concat([charts_pivoted_median, charts_pivoted_iqr, charts_pivoted_min, charts_pivoted_max], axis=1)

	#
	print(dt.now().time(), 'correcting issues with tobacco use, weight, and binary features...')
	# in tobacco use, there are only nan and non-zero values which means some percentage of the nans are actually 0
	# we hope that the percentage of actual missing data is low enough that treating all nans as 0 is a good enough approximation
	# this is the best option we have either way, as we can not know for sure after the fact
	charts_concat['tobacco_use'] = charts_concat['tobacco_use'].fillna(0)

	# delete unnecessary iqr, min and max data for tobacco usage and daily weight
	charts_concat = charts_concat.drop(['daily_weight_iqr', 'daily_weight_min', 'daily_weight_max', 'tobacco_use_iqr', 'tobacco_use_min', 'tobacco_use_max'], axis=1)

	# drop iqr, min and max columns for variables that have 2 or less unique values (not including NaN or Inf), because they're meaningless that case
	for col in charts_features.values():
		if len(np.unique(charts_concat[col])[np.isfinite(np.unique(charts_concat[col]))]) <= 2:
			before = charts_concat.shape
			charts_concat = charts_concat.drop(charts_concat.columns.intersection([f'{col}_iqr', f'{col}_min', f'{col}_max']), axis=1)
			if charts_concat.shape != before: print(dt.now().time(), 'iqr, min and max columns removed for feature:', col)

	#
	print(dt.now().time(), 'dropping (almost) completely empty rows (there should only be a few)...')
	charts_concat_after_drop_if_any = charts_concat.dropna(thresh=int(0.26*len(charts_concat.columns)), axis=0) # 26% because iqr defaults to 0, which is 25% + tobacco is filled with zeroes so this is approximately the actual threshold
	print(dt.now().time(), (f'number of rows that were almost completely empty and got dropped: {charts_concat.shape[0] - charts_concat_after_drop_if_any.shape[0]}' if charts_concat.shape != charts_concat_after_drop_if_any.shape else 'no rows were completely empty'))

	#
	print(dt.now().time(), 'KNN imputing missing data, this will take a while (it was ~20 hours for us on an Apple M1)...')
	knn_imputer = KNNImputer(n_neighbors=5)
	imputed_charts = knn_imputer.fit_transform(charts_concat_after_drop_if_any)
	charts_final = pd.DataFrame(imputed_charts, columns=charts_concat_after_drop_if_any.columns, index=charts_concat_after_drop_if_any.index)

	#
	print(dt.now().time(), 'done, saving \'1_charts\'...')
	dd.to_parquet(dd.from_pandas(charts_final), os.path.join(data_dir, '1_charts'), write_index=True)

	icu_charts = None
	charts = None
	charts_pivoted_median = None
	charts_pivoted_iqr  = None
	charts_pivoted_min  = None
	charts_pivoted_max  = None
	charts_concat = None
	charts_concat_after_drop_if_any = None
	charts_final = None

# ================================================================================================
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ================================================================================================

hosp_files.append('prescriptions')

def step_two_prescriptions():
	print(dt.now().time(), '2. prescriptions')
	prescriptions = dd.read_csv(os.path.join(hosp_dir, 'prescriptions.csv'), dtype = {
		'subject_id': 'int64',
		'hadm_id': 'int64',
		'pharmacy_id': 'Int64',
		'poe_id': 'Int64',
		'poe_seq': 'Int64',
		'order_provider_id': 'string',
		'starttime': 'string',
		'stoptime': 'string',
		'drug_type': 'string',
		'drug': 'string',
		'formulary_drug_cd': 'string',
		'gsn': 'string',
		'ndc': 'string',
		'prod_strength': 'string',
		'form_rx': 'string',
		'dose_val_rx': 'string',
		'dose_unit_rx': 'string',
		'form_val_disp': 'string',
		'form_unit_disp': 'string',
		'doses_per_24_hrs': 'Float64',
		'route': 'string'
	}, usecols = [
		'hadm_id',
		'starttime',
		'stoptime',
		'drug',
		'dose_val_rx',
		'doses_per_24_hrs',
	])

	prescriptions_features = {
		'epoetin': [],
		'warfarin': [],
		'heparin': [],
		'enoxaparin': [],
		'fondaparinux': [],
		'aspirin': [],
		'keterolac': [],
		'acetaminophen': [],
		'insulin': [],
		'glucagon': [],
		'potassium': [],
		'calcium gluconate': [],
		'fentanyl': [],
		'magnesium sulfate': [],
		'D5W': [],
		'dextrose': [],
		'ranitidine': [],
		'ondansetron': [],
		'pantoprazole': [],
		'metoclopramide': [],
		'lisinopril': [],
		'captopril': [],
		'statin': [],
		'hydralazine': [],
		'diltiazem': [],
		'carvedilol': [],
		'metoprolol': [],
		'labetalol': [],
		'atenolol': [],
		'amiodarone': [],
		'digoxin': [],
		'clopidogrel': [],
		'nitroprusside': [],
		'nitroglycerin': [],
		'vasopressin': [],
		'hydrochlorothiazide': [],
		'furosemide': [],
		'atropine': [],
		'neostigmine': [],
		'levothyroxine': [],
		'oxycodone': [],
		'hydromorphone': [],
		'fentanyl citrate': [],
		'tacrolimus': [],
		'prednisone': [],
		'phenylephrine': [],
		'norepinephrine': [],
		'haloperidol': [],
		'phenytoin': [],
		'trazodone': [],
		'levetiracetam': [],
		'diazepam': [],
		'clonazepam': [],
		'propofol': [],
		'zolpidem': [],
		'midazolam': [],
		'albuterol': [],
		'ipratropium': [],
		'diphenhydramine': [],
		'sodium chloride': [],
		'phytonadione': [],
		'metronidazole': [],
		'cefazolin': [],
		'cefepime': [],
		'vancomycin': [],
		'levofloxacin': [],
		'cipfloxacin': [],
		'fluconazole': [],
		'meropenem': [],
		'ceftriaxone': [],
		'piperacillin': [],
		'ampicillin-sulbactam': [],
		'nafcillin': [],
		'oxacillin': [],
		'amoxicillin': [],
		'penicillin': [],
		'sulfamethoxazole': [],
	}

	#
	print(dt.now().time(), 'mapping drugs of interest from prescriptions...')
	for drug in tqdm(prescriptions['drug'].dropna().unique()):
		for key in prescriptions_features.keys():
			if str(key).lower() in drug.lower():
				if 'desensit' in drug.lower() or 'graded challenge' in drug.lower(): continue # these are not used for treatment
				match str(key).lower():
					case 'diphenhydramine':
						if 'chewable' in drug.lower() or '1% cream' in drug.lower():
							continue
					case 'sodium chloride':
						if not re.match(r'(?i)^(sodium\s*chloride\s*0\.9\s*%\s*|0\.9\s*%\s*sodium\s*chloride)$', drug.lower(), re.IGNORECASE):
							continue
					case 'metronidazole':
						if '%' in drug.lower():
							continue
					case 'cefazolin': 
						if not re.match(r'^\s*cefazolin\s*$', drug.lower(), re.IGNORECASE):
							continue
					case _: pass
				prescriptions_features[key].append(drug)

	#
	print(dt.now().time(), 'filtering to drugs of interest...')
	reverse_mapping = {drug: group for group, drugs in prescriptions_features.items() for drug in drugs}

	def map_and_filter(drug):
		return reverse_mapping.get(drug, None)

	prescriptions_filtered = prescriptions.assign(mapped_drug=prescriptions['drug'].map(map_and_filter, meta=('drug', 'string')))
	prescriptions_filtered = prescriptions_filtered[prescriptions_filtered['mapped_drug'].notnull()]
	prescriptions_filtered = prescriptions_filtered.drop(columns=['drug'])

	#
	print(dt.now().time(), 'calculating total daily doses...')
	def safe_convert_and_multiply(x, y):
		def to_numeric(s):
			return pd.to_numeric(s, errors='coerce').fillna(1)

		x_numeric = to_numeric(x)
		y_numeric = to_numeric(y)
		return x_numeric * y_numeric

	prescriptions_filtered['value'] = prescriptions_filtered.map_partitions(
		lambda df: safe_convert_and_multiply(
			df['dose_val_rx'],
			df['doses_per_24_hrs']
		),
		meta = ('value', 'float64')
	)

	prescriptions_filtered = prescriptions_filtered.drop(columns=['dose_val_rx', 'doses_per_24_hrs'])
	prescriptions_filtered = prescriptions_filtered.drop_duplicates().dropna()

	#
	print(dt.now().time(), 'expanding day ranges...')
	prescriptions_filtered['startday'] = prescriptions_filtered['starttime'].astype('str').str.split(' ').apply(lambda x: x[0], meta=('starttime', 'string'))
	prescriptions_filtered['stopday'] = prescriptions_filtered['stoptime'].astype('str').str.split(' ').apply(lambda x: x[0], meta=('stoptime', 'string'))
	prescriptions_filtered = prescriptions_filtered.drop(columns=['starttime', 'stoptime'])

	# there are some rows where the stopday is before the startday.
	# we will mark these as only one day, as the only reasonable explanation that we could come up with
	# is that the medication was only taken for one day and this is how they marked it.
	# this might be wrong but we still believe it is better than swapping the days, as that could introduce larger errors.
	mask = prescriptions_filtered['startday'] > prescriptions_filtered['stopday']
	prescriptions_filtered['stopday'] = prescriptions_filtered['startday'].where(mask, prescriptions_filtered['stopday'])

	def expand_date_range(df):
		df['startday'] = pd.to_datetime(df['startday'])
		df['stopday'] = pd.to_datetime(df['stopday'])
		df['days'] = df.apply(lambda row: pd.date_range(row['startday'], row['stopday'], freq='D'), axis=1)
		df = df.explode('days')
		df = df.rename(columns={'days': 'day'}).drop(columns=['startday', 'stopday'])
		df.astype({'day': 'string'})
		return df

	prescriptions_expanded = expand_date_range(prescriptions_filtered.compute())
	prescriptions_expanded['HADMID_DAY'] = prescriptions_expanded['hadm_id'].astype('str') + '_' + prescriptions_expanded['day'].astype('str')

	#
	print(dt.now().time(), 'creating daily aggregates...')
	prescriptions_pivoted = pd.pivot_table(prescriptions_expanded, index='HADMID_DAY', columns='mapped_drug', values='value', aggfunc=np.amax, fill_value=0, dropna=False)
	prescriptions_final = dd.from_pandas(prescriptions_pivoted)

	#
	print(dt.now().time(), 'merging with previous step...')
	charts = dd.read_parquet(os.path.join(data_dir, '1_charts'), index='HADMID_DAY')
	charts_prescriptions = dd.merge(charts, prescriptions_final, on='HADMID_DAY', how='left')
	charts_prescriptions = charts_prescriptions.fillna(0)

	#
	print(dt.now().time(), 'done, saving \'2_charts_prescriptions\'...')
	dd.to_parquet(charts_prescriptions, os.path.join(data_dir, '2_charts_prescriptions'), write_index=True)

	prescriptions = None
	prescriptions_filtered = None
	prescriptions_expanded = None
	prescriptions_pivoted = None
	prescriptions_final = None
	charts = None
	charts_prescriptions = None

# ================================================================================================
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ================================================================================================

hosp_files.append('admissions')

def step_three_admissions():
	print(dt.now().time(), '3. admissions')
	admissions = dd.read_csv(os.path.join(hosp_dir, 'admissions.csv'), dtype = {
		'subject_id': 'int64',
		'hadm_id': 'int64',
		'admittime': 'string',
		'dischtime': 'string',
		'deathtime': 'string',
		'admission_type': 'string',
		'admit_provider_id': 'string',
		'admission_location': 'string',
		'discharge_location': 'string',
		'insurance': 'string',
		'language': 'string',
		'marital_status': 'string',
		'race': 'string',
		'edregtime': 'string',
		'edouttime': 'string',
		'hospital_expire_flag': 'string'
	}, usecols = [
		'subject_id',
		'hadm_id',
		'deathtime',
		'race',
		'hospital_expire_flag',
	])

	#
	print(dt.now().time(), 'mapping ethnicity...') 
	admissions['is_black'] = admissions['race'].apply(
		lambda x: 1 if 'BLACK' in str(x).upper() else 0,
		meta=('race_binary', 'int8')
	)

	#
	print(dt.now().time(), 'mapping to HADMID_DAY...')
	charts_prescriptions = dd.read_parquet(os.path.join(data_dir, '2_charts_prescriptions'), index='HADMID_DAY')

	hadmid_day = charts_prescriptions.index.to_frame()
	hadmid_day['hadm_id'] = hadmid_day['HADMID_DAY'].str.split('_').str[0]
	hadmid_day['day'] = hadmid_day['HADMID_DAY'].str.split('_').str[1]
	hadmid_day = hadmid_day.astype({
		'HADMID_DAY': 'string',
		'hadm_id': 'int64',
		'day': 'string',
	})

	hadmid_day_admissions = hadmid_day.merge(admissions, on='hadm_id', how='left')

	#
	print(dt.now().time(), 'merging with previous step...')
	charts_prescriptions_admissions = dd.merge(charts_prescriptions, admissions_final, on='HADMID_DAY', how='left')
	hadmid_day_admissions = hadmid_day_admissions.fillna(0)

	#
	print(dt.now().time(), 'done, saving \'3_charts_prescriptions_admissions\'...')
	dd.to_parquet(charts_prescriptions_admissions, os.path.join(data_dir, '3_charts_prescriptions_admissions'), write_index=True)

	admissions = None
	charts_prescriptions = None
	hadmid_day = None
	hadmid_day_admissions = None
	admissions_final = None
	charts_prescriptions_admissions = None

# ================================================================================================
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ================================================================================================

hosp_files.append('patients')

def step_four_patients():
	print(dt.now().time(), '4. patients')
	patients = dd.read_csv(os.path.join(hosp_dir, 'patients.csv'), dtype = {
		'subject_id': 'int64',
		'gender': 'string',
		'anchor_age': 'int64',
		'anchor_year': 'int64',
		'anchor_year_group': 'string',
		'dod': 'string'
	}, usecols = [
		'subject_id',
		'gender',
		'anchor_age',
	])

	patients = patients.compute()

	#
	print(dt.now().time(), 'mapping patients from subject_id to hadm_id as patients table does not contain hadm_id...')
	admissions = dd.read_csv(os.path.join(hosp_dir, 'admissions.csv'), dtype = { 'subject_id': 'int64', 'hadm_id': 'int64' }, usecols = [ 'subject_id', 'hadm_id' ])
	hadm_to_subject = admissions.set_index('hadm_id')['subject_id'].compute().to_dict()

	patients_hadm = pd.DataFrame(columns=patients.columns)
	for hadm in hadm_to_subject.keys():
		new_row = patients[patients['subject_id'] == hadm_to_subject[hadm]].copy()
		new_row['hadm_id'] = hadm
		patients_hadm = pd.concat([patients_hadm, new_row], ignore_index=True)

	patients_hadm = patients_hadm.astype({'hadm_id': 'int64'}).set_index('hadm_id')
	patients_hadm = patients_hadm.drop(columns=['subject_id'])

	#
	print(dt.now().time(), 'one-hot encoding gender...')
	patients_final = pd.get_dummies(patients_hadm, columns=['gender'], prefix='gender', dtype='int8')

	#
	print(dt.now().time(), 'merging with previous step...')
	charts_prescriptions_admissions = dd.read_parquet(os.path.join(data_dir, '3_charts_prescriptions_admissions'))
	charts_prescriptions_admissions['hadm_id'] = charts_prescriptions_admissions['HADMID_DAY'].str.split('_').str[0].astype('int64')
	charts_prescriptions_admissions_patients = dd.merge(charts_prescriptions_admissions, patients_final, on='hadm_id', how='left')
	charts_prescriptions_admissions_patients = charts_prescriptions_admissions_patients.fillna(0)
	charts_prescriptions_admissions_patients = charts_prescriptions_admissions_patients.drop(columns=['hadm_id'])
	charts_prescriptions_admissions_patients.set_index('HADMID_DAY')

	#
	print(dt.now().time(), 'done, saving \'4_charts_prescriptions_admissions_patients\'...')
	dd.to_parquet(charts_prescriptions_admissions_patients, os.path.join(data_dir, '4_charts_prescriptions_admissions_patients'), write_index=True)

	patients = None
	admissions = None
	hadm_to_subject = None
	patients_hadm = None
	patients_final = None
	charts_prescriptions_admissions = None
	charts_prescriptions_admissions_patients = None

# ================================================================================================
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ================================================================================================

hosp_files.append('microbiologyevents')

def step_five_microbio():
	print(dt.now().time(), '5. microbiology')
	microbio_events = dd.read_csv(os.path.join(hosp_dir, 'microbiologyevents.csv'), dtype = {
		'microevent_id': 'int64',
		'subject_id': 'int64',
		'hadm_id': 'Int64',
		'micro_specimen_id': 'int64',
		'order_provider_id': 'string',
		'chartdate': 'string',
		'charttime': 'string',
		'spec_itemid': 'int64',
		'spec_type_desc': 'string',
		'test_seq': 'int64',
		'storedate': 'string',
		'storetime': 'string',
		'test_itemid': 'Int64',
		'test_name': 'string',
		'org_itemid': 'Int64',
		'org_name': 'string',
		'isolate_num': 'Int64',
		'quantity': 'string',
		'ab_itemid': 'Int64',
		'ab_name': 'string',
		'dilution_text': 'string',
		'dilution_comparison': 'string',
		'dilution_value': 'Float64',
		'interpretation': 'category',
		'comment': 'string'
	}, usecols = [
		'microevent_id',
		'subject_id',
	])

	#
	print(dt.now().time(), 'marking patients with suspected infection...')
	# hadm_id is frequently missing, so we map by subject_id instead using the admissions table
	admissions = dd.read_csv(os.path.join(hosp_dir, 'admissions.csv'), dtype = { 'subject_id': 'int64', 'hadm_id': 'int64' }, usecols = [ 'subject_id', 'hadm_id' ])
	hadm_to_subject = admissions.astype({'hadm_id': 'int64', 'subject_id': 'int64'}).set_index('hadm_id')['subject_id'].compute().to_dict()
	suspected_subject_ids = set(pd.unique(microbio_events['subject_id'].compute()))

	#
	print(dt.now().time(), 'mapping onto previous step...')
	charts_prescriptions_admissions_patients = dd.read_parquet(os.path.join(data_dir, '4_charts_prescriptions_admissions_patients'), index='HADMID_DAY')
	charts_prescriptions_admissions_patients['hadm_id'] = charts_prescriptions_admissions_patients.index.str.split('_').str[0].astype('int64')
	charts_prescriptions_admissions_patients['suspected_infection'] = 0
	charts_prescriptions_admissions_patients = charts_prescriptions_admissions_patients.compute()

	count_infection = 0
	for index, row in charts_prescriptions_admissions_patients.iterrows():
		if int(row['hadm_id']) in hadm_to_subject.keys() and int(hadm_to_subject[int(row['hadm_id'])]) in suspected_subject_ids:
			charts_prescriptions_admissions_patients.loc[index, 'suspected_infection'] = 1
			count_infection += 1
	print(dt.now().time(), 'patients mapped with suspected infection:', count_infection)

	charts_prescriptions_admissions_patients_microbio = charts_prescriptions_admissions_patients.drop(columns=['hadm_id'])

	#
	print(dt.now().time(), 'done, saving \'5_charts_prescriptions_admissions_patients_microbio\'...')
	dd.to_parquet(dd.from_pandas(charts_prescriptions_admissions_patients_microbio), os.path.join(data_dir, '5_charts_prescriptions_admissions_patients_microbio'), write_index=True)

	microbio_events = None
	admissions = None
	suspected_subject_ids = None
	charts_prescriptions_admissions_patients = None
	charts_prescriptions_admissions_patients_microbio = None

# ================================================================================================
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ================================================================================================

hosp_files.append('diagnoses_icd')

def step_six_diagnoses():
	print(dt.now().time(), '6. diagnoses')
	diagnoses = dd.read_csv(os.path.join(hosp_dir, 'diagnoses_icd.csv'), dtype = {
		'subject_id': 'int64',
		'hadm_id': 'int64',
		'seq_num': 'int64',
		'icd_code': 'string',
		'icd_version': 'int8',
	}, usecols = [
		'hadm_id',
		'icd_code',
		'icd_version',
	])

	#
	print(dt.now().time(), 'marking patients with chronic kidney disease...')
	ckd_hadmids = set(pd.unique(diagnoses[(diagnoses['icd_code'].str.startswith('585') & (diagnoses['icd_version'] == 9)) | (diagnoses['icd_code'].str.startswith('N18') & diagnoses['icd_version'] == 10)]['hadm_id'].astype({'hadm_id': 'int64'}).compute()))

	#
	print(dt.now().time(), 'mapping onto previous step...')
	charts_prescriptions_admissions_patients_microbio = dd.read_parquet(os.path.join(data_dir, '5_charts_prescriptions_admissions_patients_microbio'), index='HADMID_DAY')

	charts_prescriptions_admissions_patients_microbio['ckd'] = 0
	charts_prescriptions_admissions_patients_microbio['hadm_id'] = charts_prescriptions_admissions_patients_microbio.index.str.split('_').str[0].astype('int64')
	charts_prescriptions_admissions_patients_microbio = charts_prescriptions_admissions_patients_microbio.compute()

	count_ckd = 0
	for index, row in charts_prescriptions_admissions_patients_microbio.iterrows():
		if int(row['hadm_id']) in ckd_hadmids:
			charts_prescriptions_admissions_patients_microbio.loc[index, 'ckd'] = 1
			count_ckd += 1
	print(dt.now().time(), 'patients mapped with chronic kidney disease:', count_ckd)

	charts_prescriptions_admissions_patients_microbio_diagnoses = charts_prescriptions_admissions_patients_microbio.drop(columns=['hadm_id'])
	
	#
	print(dt.now().time(), 'done, saving \'6_charts_prescriptions_admissions_patients_microbio_diagnoses\'...')
	dd.to_parquet(dd.from_pandas(charts_prescriptions_admissions_patients_microbio_diagnoses), os.path.join(data_dir, '6_charts_prescriptions_admissions_patients_microbio_diagnoses'), write_index=True)

	diagnoses = None
	ckd_hadmids = None
	charts_prescriptions_admissions_patients_microbio = None
	charts_prescriptions_admissions_patients_microbio_diagnoses = None

# ================================================================================================
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ================================================================================================

if __name__ == "__main__":
	print(dt.now().time(), 'MIMIC-IV data cleaning pipeline - expect this to take up to a day or two to run depending on your hardware')
	ungzip_required(hosp_dir, hosp_files)
	ungzip_required(icu_dir, icu_files)

	# each step depends on the output of the previous one being present in the data_dir.
	# if you are modifying the code and you need to quickly debug/iterate on a single step, 
	# run everything up to it and then comment out the calls to rest of them below.

	step_one_charts()
	step_two_prescriptions()
	step_three_admissions()
	step_four_patients()
	step_five_microbio()
	step_six_diagnoses()