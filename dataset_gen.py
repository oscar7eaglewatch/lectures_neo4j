from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import numpy as np
import os

ALGORITHSET = 1 # 1, 2, 3

train_dataset_path = os.path.join(os.getcwd(), str (ALGORITHSET) + '-train.npy')
valid_dataset_path = os.path.join(os.getcwd(), str (ALGORITHSET) + '-valid.npy')
test_dataset_path = os.path.join(os.getcwd(), str (ALGORITHSET) + '-test.npy')

train_list = np.load(train_dataset_path)
valid_list = np.load(valid_dataset_path)
test_list = np.load(test_dataset_path)


train_text = []
train_summary = []
for each in train_list:
    train_text.append(each.split("<diaries>")[0])
    train_summary.append(each.split("<diaries>")[1].replace("<final_thi_score>", ""))   

valid_text = []
valid_summary = []
for each in valid_list:
    valid_text.append(each.split("<diaries>")[0])
    valid_summary.append(each.split("<diaries>")[1].replace("<final_thi_score>", ""))   

test_text = []
test_summary = []
for each in test_list:
    test_text.append(each.split("<diaries>")[0])
    test_summary.append(each.split("<diaries>")[1].replace("<final_thi_score>", ""))   



d = {'train': Dataset.from_dict({'summary': train_summary, 'text': train_text}),
     'valid': Dataset.from_dict({'summary': valid_summary, 'text': valid_text}),
     'test': Dataset.from_dict({'summary': test_summary, 'text': test_text})}

dataset = DatasetDict(d)

dataset.save_to_disk(os.path.join(os.getcwd(), 'augmented-dataset-' + str(ALGORITHSET)))