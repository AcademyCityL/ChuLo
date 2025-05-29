# -*- coding: utf-8 -*-
import os
import pandas as pd

def process_20ng(root_path):
	data_frame= {'document':[],'split':[],'label':[],'filename':[]}
	splits = ["20news-bydate-test","20news-bydate-train"]
	for split in splits:
		split_path = os.path.join(root_path,split)
		item_list = os.listdir(split_path)
		# print("item_list ",item_list)
		for item in item_list:
			if item == '.DS_Store':
				continue
			next_path = os.path.join(split_path, item)
			# print(next_path,item)
			file_list = os.listdir(next_path)
			# print("file_list ",file_list)
			for file in file_list:
				if file == '.DS_Store':
					continue
				# print(file)
				file_path = os.path.join(next_path,file)
				# print(file_path)
				with open(file_path, "r", encoding="latin1") as f:
					content = f.read()
					content = content.strip()
					data_frame['document'].append(content)
					data_frame['split'].append(split)
					data_frame['label'].append(item)
					data_frame['filename'].append(file)
	df = pd.DataFrame(data_frame)
	df.to_csv('20ng.csv', index=False, header=True)


process_20ng('/Users/liyan/Documents/GitHub/text_gcn/data/20ng')



