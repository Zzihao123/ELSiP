import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):


		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = '.'.join(self.slide_data['slide_id'][idx].split('.')[:-1])
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir
		self.use_h5 = True
		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, '{}.pt'.format(slide_id))
				features = torch.load(full_path)
				return features, label
			
			else:
				return slide_id, label

		else:
			full_path = os.path.join(data_dir, '{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, coords


class Generic_MIL_Dataset_2(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset_2, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = '.'.join(self.slide_data['slide_id'][idx].split('.')[:-1])
		label = self.slide_data['label'][idx]
		sex = self.slide_data['sex'][idx]
		age = self.slide_data['age'][idx]
		other = self.slide_data['other'][idx]
		
		# Convert demographic and clinical information to formatted text
		clinical_text = self.format_clinical_info(sex, age, other)
		
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir
		self.use_h5 = True
		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, '{}.pt'.format(slide_id))
				features = torch.load(full_path)
				return features, label, clinical_text
			
			else:
				return slide_id, label, clinical_text

		else:
			full_path = os.path.join(data_dir, '{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, clinical_text, coords

	def format_clinical_info(self, sex, age, other):
		"""
		Convert demographic and clinical information into formatted English text.
		
		Args:
			sex (str): Patient sex ('m' or 'f')
			age (int): Patient age
			other (str): Clinical description of the eye condition
			
		Returns:
			str: Formatted clinical description
		"""
		# Convert sex to full form
		sex_full = "male" if sex == "m" else "female"
		
		# Clean and format the clinical description
		clinical_desc = other.strip()
		
		# Create a well-formatted clinical description
		formatted_text = f"Patient is a {age}-year-old {sex_full}. Clinical presentation: {clinical_desc}."
		
		return formatted_text


class Generic_MultiModal_Dataset(Generic_WSI_Classification_Dataset):
	"""
	Multi-modal dataset supporting slide features, patch features, and clinical text information.
	"""
	def __init__(self,
		slide_data_dir, 
		patch_data_dir=None,
		**kwargs):
	
		super(Generic_MultiModal_Dataset, self).__init__(**kwargs)
		self.slide_data_dir = slide_data_dir
		self.patch_data_dir = patch_data_dir
		self.use_h5 = True  # Default to h5 format

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = '.'.join(self.slide_data['slide_id'][idx].split('.')[:-1])
		label = self.slide_data['label'][idx]
		
		# Extract clinical information and format as text
		clinical_text = ""
		if 'sex' in self.slide_data.columns and 'age' in self.slide_data.columns and 'other_translated' in self.slide_data.columns:
			sex = self.slide_data['sex'][idx]
			age = self.slide_data['age'][idx]
			other = self.slide_data['other_translated'][idx]
			clinical_text = self.format_clinical_info(sex, age, other)
		
		# Load slide features
		slide_features,slide_coords = self.load_slide_features(slide_id)
		
		# Load patch features if available
		patch_features = None
		patch_coords = None
		if self.patch_data_dir:
			patch_features, patch_coords = self.load_patch_features(slide_id)
		
		if not self.use_h5:
			if self.slide_data_dir:
				# Return features and clinical text
				if patch_features is not None:
					return slide_features, patch_features, label, clinical_text
				else:
					return slide_features, label, clinical_text
			else:
				return slide_id, label, clinical_text
		else:
			# H5 format
			if patch_features is not None:
				return slide_features, patch_features, label, clinical_text, slide_coords, patch_coords
			else:
				return slide_features, label, clinical_text, slide_coords

	def load_slide_features(self, slide_id):
		"""Load slide-level features"""
		if type(self.slide_data_dir) == dict:
			# Handle multiple data sources
			slide_features = []
			slide_coords = []
			for source, data_dir in self.slide_data_dir.items():
				full_path = os.path.join(data_dir, '{}.h5'.format(slide_id))
				if os.path.exists(full_path):
					with h5py.File(full_path,'r') as hdf5_file:
						features = hdf5_file['features'][:]
						coords = hdf5_file['coords'][:]
					slide_features.append(torch.from_numpy(features))
					slide_coords.append(coords)
			if slide_features:
				combined_features = torch.cat(slide_features, dim=0) if len(slide_features) > 1 else slide_features[0]
				combined_coords = np.concatenate(slide_coords, axis=0) if len(slide_coords) > 1 else slide_coords[0]
				return combined_features, combined_coords
			else:
				raise FileNotFoundError(f"Slide features not found for {slide_id}")
		else:
			full_path = os.path.join(self.slide_data_dir, '{}.h5'.format(slide_id))
			if os.path.exists(full_path):
				with h5py.File(full_path,'r') as hdf5_file:
					features = hdf5_file['features'][:]
					coords = hdf5_file['coords'][:]

				features = torch.from_numpy(features)
				return features, coords
			else:
				raise FileNotFoundError(f"Slide features not found at {full_path}")

	def load_patch_features(self, slide_id):
		"""Load patch-level features"""
		if type(self.patch_data_dir) == dict:
			# Handle multiple patch data sources
			patch_features = []
			patch_coords = []
			for source, data_dir in self.patch_data_dir.items():
				full_path = os.path.join(data_dir, '{}.h5'.format(slide_id))
				if os.path.exists(full_path):
					with h5py.File(full_path,'r') as hdf5_file:
						features = hdf5_file['features'][:]
						coords = hdf5_file['coords'][:]
					patch_features.append(torch.from_numpy(features))
					patch_coords.append(coords)
			if patch_features:
				combined_features = torch.cat(patch_features, dim=0) if len(patch_features) > 1 else patch_features[0]
				combined_coords = np.concatenate(patch_coords, axis=0) if len(patch_coords) > 1 else patch_coords[0]
				return combined_features, combined_coords
			else:
				return None, None
		else:
			full_path = os.path.join(self.patch_data_dir, '{}.h5'.format(slide_id))
			if os.path.exists(full_path):
				with h5py.File(full_path,'r') as hdf5_file:
					features = hdf5_file['features'][:]
					coords = hdf5_file['coords'][:]
				features = torch.from_numpy(features)
				return features, coords
			else:
				return None, None

	def format_clinical_info(self, sex, age, other):
		"""
		Convert demographic and clinical information into formatted English text.
		
		Args:
			sex (str): Patient sex ('M' or 'F')
			age (int): Patient age
			other (str): Clinical description of the eye condition
			
		Returns:
			str: Formatted clinical description
		"""
		# Convert sex to full form
		sex_full = "male" if sex.upper() == "M" else "female"
		
		# Clean and format the clinical description
		clinical_desc = other.strip()
		
		# Create a well-formatted clinical description
		formatted_text = f"Patient is a {age}-year-old {sex_full}. Clinical presentation: {clinical_desc}."
		# formatted_text = f"Patient is a {age}-year-old {sex_full}."

		return formatted_text

	def get_feature_dimensions(self):
		"""Get the dimensions of slide and patch features"""
		if len(self) == 0:
			return None, None
		
		# Get first sample to determine dimensions
		sample = self[0]
		if len(sample) == 4:  # slide, patch, label, clinical_text
			slide_dim = sample[0].shape[-1] if hasattr(sample[0], 'shape') else None
			patch_dim = sample[1].shape[-1] if hasattr(sample[1], 'shape') else None
		elif len(sample) == 3:  # slide, label, clinical_text
			slide_dim = sample[0].shape[-1] if hasattr(sample[0], 'shape') else None
			patch_dim = None
		else:
			slide_dim = None
			patch_dim = None
			
		return slide_dim, patch_dim

	def return_splits(self, from_id=True, csv_path=None):
		"""
		Override return_splits to use Generic_MultiModal_Split
		"""
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_MultiModal_Split(train_data, slide_data_dir=self.slide_data_dir, patch_data_dir=self.patch_data_dir, num_classes=self.num_classes)
			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_MultiModal_Split(val_data, slide_data_dir=self.slide_data_dir, patch_data_dir=self.patch_data_dir, num_classes=self.num_classes)
			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_MultiModal_Split(test_data, slide_data_dir=self.slide_data_dir, patch_data_dir=self.patch_data_dir, num_classes=self.num_classes)
			else:
				test_split = None
			
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_MultiModal_Split(df_slice, slide_data_dir=self.slide_data_dir, patch_data_dir=self.patch_data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split


class Generic_MultiModal_Split(Generic_MultiModal_Dataset):
	def __init__(self, slide_data, slide_data_dir=None, patch_data_dir=None, num_classes=2):
		self.use_h5 = True
		self.slide_data = slide_data
		self.slide_data_dir = slide_data_dir
		self.patch_data_dir = patch_data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)



