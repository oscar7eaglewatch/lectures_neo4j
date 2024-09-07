import os
import tqdm
import torch
import random
import numpy as np
import nlpaug.augmenter.word as naw

cuda_available = torch.cuda.is_available()

print("Is CUDA available? :", cuda_available)

if cuda_available:
    deviceName = "cuda"
else:
    deviceName = "cpu"

loaded_list = np.load('tinnitus_collected_v4.npy')

GEN_MODE = 'TRAIN' # 'VALID' or 'TEST' case sensitivce
ALGORITHSET = 1    # 1, 2, 3


if GEN_MODE == 'TRAIN':
    repeat = 300
    filename = os.path.join(os.getcwd(), str (ALGORITHSET) + '-train.npy')
elif GEN_MODE == 'VALID':
    repeat = 4
    filename = os.path.join(os.getcwd(), str (ALGORITHSET) + '-valid.npy')
elif GEN_MODE == 'TEST':
    repeat = 2
    filename = os.path.join(os.getcwd(), str (ALGORITHSET) + '-test.npy')
else:
    print ("error!, GEN_MODE must be either TRAIN, or VALID, or TEST! Case sensitive.")

if ALGORITHSET == 1 or ALGORITHSET == 2:
    e_rate = 0.3
    e_pct = 40
elif ALGORITHSET == 3:
    e_rate = 0.5
    e_pct = 50
else:
    print ("error!, ALGORITHSET an integer range from 1 to 3.")


def get_augmentation(augmentation_set=1, devicetype="cuda"):
    if augmentation_set==1:
        augmentations = {}
        augmentations["synonym_replace1"] = naw.SynonymAug(aug_src='wordnet', aug_min=10, aug_max=20)
        augmentations["random_substitute"] = naw.ContextualWordEmbsAug(model_path="distilbert-base-uncased", device=devicetype, action="substitute", aug_p=0.5, top_k=10)
        augmentations["synonym_replace2"] = naw.SynonymAug(aug_src='wordnet', aug_min=5, aug_max=10)
    elif augmentation_set==2:
        augmentations = {}
        augmentations["synonym_replace1"] = naw.SynonymAug(aug_src='wordnet', aug_min=10, aug_max=20)
        augmentations["random_substitute"] = naw.ContextualWordEmbsAug(model_path="roberta-base", device=devicetype, action="substitute", aug_p=0.9, top_k=20)
        augmentations["synonym_replace2"] = naw.SynonymAug(aug_src='wordnet', aug_min=5, aug_max=10)
        augmentations["random_swap"] = naw.RandomWordAug(action="swap")
        augmentations["synonym_replace3"] = naw.SynonymAug(aug_src='wordnet', aug_min=10, aug_max=10)
    elif augmentation_set==3:
        augmentations = {}
        augmentations["random_swap"] = naw.RandomWordAug(action="swap")
        augmentations["random_substitute"] = naw.ContextualWordEmbsAug(model_path="roberta-base", device=devicetype, action="substitute", aug_p=0.7, top_k=30)
        augmentations["synonym_replace2"] = naw.SynonymAug(aug_src='wordnet', aug_min=5, aug_max=10)
        augmentations["synonym_replace1"] = naw.SynonymAug(aug_src='wordnet', aug_min=10, aug_max=20)
    else:
        print ("error!")
        return []
    return augmentations


def continuous_augumentation(augmentations, original_text, augmentation_set=1):
    words = len(original_text.split(" "))
    if words == 1:
        text = augmentations["synonym_replace1"].augment(original_text)[0]
    else:
        text = original_text
        for k,v in augmentations.items():
            if augmentation_set == 3:
                if k != "synonym_replace1":
                    text = v.augment(text)[0]
            else:
                text = v.augment(text)[0]

    return text

def bootstrap_sample_with_error(value, error_rate, error_percentage, total_samples):
    samples = []
    for _ in range(total_samples):  # Change 1000 to whatever number of samples you want
        if random.random() < error_rate:
            error_amount = value * (error_percentage / 100)
            sample = value + random.uniform(-error_amount, error_amount)
            samples.append(sample)
        else:
            samples.append(value)
    return sum(samples) / len(samples)  # Return the mean of the samples

def get_string_data_boostrapping_sampling(data_string, error_rate=0.3, error_percentage=20, total_samples=5):
    #data_str = "25.0 21.0 22.0 20.0"
    data_list = [float(item) for item in data_string.split()]

    # List to store the mean of the bootstrapped samples for each float value
    bootstrapped_means = [bootstrap_sample_with_error(item, error_rate, error_percentage, total_samples) for item in data_list]
    
    # Convert the results back to string format
    result_str = " ".join(f"{item:.2f}" for item in bootstrapped_means)
    return result_str

augmentations = get_augmentation(augmentation_set=ALGORITHSET, devicetype=deviceName)

augumented_list = []
for k in range(42):
    each_entry = loaded_list[k]
    # repeat = 1000 
    # for train : repeat = 1000
    # for valid : repeat = 250
    # for test  : repeat = 50 and add the original to test at the end

    # Algorithm 2 and 3 
    # e_rate = 0.3, e_pct = 40

    # Algorithm 4
    # e_rate = 0.5, e_pct = 50

    #e_rate = 0.3
    #e_pct = 40

    for each_repeat in tqdm.tqdm(range(repeat)):
        freq_info = each_entry.split("<freq_info>")[0]
        updated_freq_info = get_string_data_boostrapping_sampling(freq_info, error_rate=e_rate, error_percentage=e_pct, total_samples=5)

        initial_thi_score = each_entry.split("<initial_thi_score>")[0].split("<freq_info>")[1]

        updated_initial_thi_score = get_string_data_boostrapping_sampling(initial_thi_score, error_rate=e_rate, error_percentage=e_pct, total_samples=5)

        final_thi_score = each_entry.split("<initial_thi_score>")[1].split("<diaries>")[1].replace("<final_thi_score>", "")
        updated_final_thi_score = get_string_data_boostrapping_sampling(final_thi_score, error_rate=e_rate, error_percentage=e_pct, total_samples=5)

        diaries = each_entry.split("<initial_thi_score>")[1].split("<diaries>")[0]
        diaries_splits = diaries.split(":")

        augument_diaries = [""] * len(diaries_splits)
        for i in range(len(diaries_splits)):
            if len(diaries_splits[i]) > 0:
                augument_diaries[i] = continuous_augumentation(augmentations, diaries_splits[i], augmentation_set=ALGORITHSET)
        updated_diaries = ":".join(augument_diaries)

        augumented_list.append(updated_freq_info + "<freq_info>" + updated_initial_thi_score + "<initial_thi_score>" + updated_diaries + "<diaries>" + updated_final_thi_score +"<final_thi_score>")



np.save(filename, augumented_list)
print ("Saving completed!")