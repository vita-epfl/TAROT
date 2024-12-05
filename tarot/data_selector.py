import geomloss
import numpy as np
from tarot.utils import get_matrix_mult
import os
from multiprocessing import Pool
import torch
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Subset
import numpy as np
import gzip
import torch
from torch.utils.data import Dataset

def numba_topk_2d_axis0(args):
    arr, k = args

    # Find the partitioned indices for top-k elements along axis 0
    k = min(k, arr.shape[0])
    partitioned_indices = np.argpartition(arr, -k, axis=0)[-k:]

    # Retrieve the k largest values based on the partitioned indices
    k_largest_values = np.take_along_axis(arr, partitioned_indices, axis=0)

    # Sort the k largest values along axis 0 to get sorted indices within top-k
    sorted_indices_within_k = np.argsort(-k_largest_values, axis=0)

    # Get the sorted indices for the original array
    sorted_indices = np.take_along_axis(partitioned_indices, sorted_indices_within_k, axis=0)

    # Get the sorted scores using the sorted indices
    sorted_scores = np.take_along_axis(arr, sorted_indices, axis=0)

    return sorted_scores, sorted_indices


def multi_process_sort(array, k=2000):
    process_num = os.cpu_count()
    data_splits = np.array_split(array, process_num, axis=1)

    # Prepare arguments as tuples
    args = [(split, k) for split in data_splits]

    with Pool(processes=process_num) as pool:
        # Map function that takes a tuple of arguments
        results = pool.map(numba_topk_2d_axis0, args)

    # Separate the sorted indices and scores from the results
    sorted_indices = np.concatenate([result[1] for result in results], axis=1)
    sorted_scores = np.concatenate([result[0] for result in results], axis=1)

    return sorted_scores, sorted_indices




class DataSelector:

    def __init__(self, cfg):
        self.device = cfg['device']
        self.method = cfg['selection_method']
        self.selection_ratio = cfg['selection_ratio']
        self.k_fold_splits = cfg['k_fold_splits']
        self.merged = cfg['merge_target_data']
        self.data_weighting = cfg['data_weighting']
        self.ot_distance_computer = geomloss.SamplesLoss(
        loss='sinkhorn',
        cost=self.cosine_L2,
        debias=False,
        blur=0.01,
        potentials=False,
        backend='tensorized',
        scaling=0.5
        )
    def cosine_L2(self,x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        sim = get_matrix_mult(x[0], y[0])
        sim.mul_(-2)
        sim.add_(2)
        sim.abs_()
        sim.sqrt_()
        sim = sim.to(self.device)
        return sim.unsqueeze(0)

    def select_data(self,score, candidate_feautres, target_features):
        num_selected = int(self.selection_ratio * score.shape[0])
        method = self.method
        candidate_number, target_number = score.shape

        assert num_selected <= score.shape[0]
        if method == 'random':
            selected_train_files = np.random.choice(score.shape[0], num_selected, replace=False)
            weight = np.ones_like(selected_train_files)
        elif method == 'dsdm' or method == 'less':
            avg_score = np.mean(score, axis=1)
            closest_training_ids = np.argsort(avg_score)
            selected_train_files = closest_training_ids[-num_selected:].reshape(-1)
            weight = np.ones_like(selected_train_files)
        elif method == 'otm':
            sorted_score, sorted_indices = multi_process_sort(score)
            g = candidate_feautres.to(torch.float32).to('cpu')
            g_target = target_features.to(torch.float32).to('cpu')

            kf = KFold(n_splits=self.k_fold_splits, shuffle=True, random_state=42)
            selected_train_files = []

            for fold, (train_index, test_index) in enumerate(kf.split(np.arange(target_number))):
                print(f"Fold {fold + 1}")
                target_val = test_index
                target_test = train_index
                g_target_fold = g_target[target_test]
                min_c_fold = 1e9
                dist_threshold = 0.0
                distance = self.ot_distance_computer
                for k in range(1, sorted_score.shape[0]):
                    # Use the pre-sorted indices and scores
                    sorted_indices_k = sorted_indices[:k, target_val].reshape(-1)
                    sorted_score_k = sorted_score[:k, target_val].reshape(-1)
                    score_above_threshold = sorted_score_k > dist_threshold

                    sorted_indices_k = sorted_indices_k[score_above_threshold]

                    unique_indices, cnt = np.unique(sorted_indices_k, return_counts=True)
                    c = distance(g[unique_indices].to(self.device), g_target_fold.to(self.device))
                    print(f'k: {k}, distance: {c.item()}')
                    if c.item() < min_c_fold:
                        min_c_fold = c.item()
                    else:
                        break

                print(f'Final distance for fold {fold + 1}: {min_c_fold}')
                selected_train_files.append(unique_indices)
                print(f'Selection ratio for fold {fold + 1}: {len(unique_indices) / candidate_number}')

            # Print overall result
            selected_train_files = np.concatenate(selected_train_files)
            unique_indices, cnt = np.unique(selected_train_files, return_counts=True)
            selected_train_files = unique_indices

            distance.potentials = True
            f, _ = distance(g[unique_indices].to(self.device), g_target.to(self.device))
            weight = f[0].detach().cpu().numpy()

        elif method == 'fixed_size':
            print(f'num_selected: {num_selected}')
            g = candidate_feautres.to(torch.float32).to('cpu')
            g_target = target_features.to(torch.float32).to('cpu')
            sorted_score, sorted_indices = multi_process_sort(score)
            unique_set = set()
            distance = self.ot_distance_computer
            for i in range(sorted_indices.shape[0]):
                this_layer = np.unique(sorted_indices[i])
                updated_set = unique_set.union(this_layer)
                # count the number of unique elements
                if len(updated_set) <= num_selected:
                    unique_set = updated_set
                else:
                    number_to_add = num_selected - len(unique_set)
                    print(f'number to add: {number_to_add}')
                    diff = np.setdiff1d(this_layer, list(unique_set))

                    distance.potentials = True
                    f, _ = distance(g[diff].to(self.device), g_target.to(self.device))
                    seleted_weight = f[0].detach().cpu().numpy()
                    sorted_diff = np.argsort(seleted_weight)
                    unique_set = unique_set.union(diff[sorted_diff[-number_to_add:]])
                    break
            distance.potentials = True
            g_selected = g[list(unique_set)]
            f, _ = distance(g_selected.to(self.device), g_target.to(self.device))
            weight = f[0].detach().cpu().numpy()
            selected_train_files = np.array(list(unique_set))

        print(f'Overall selection ratio: {len(selected_train_files) / candidate_number}')
        print(f'using {method} method, select {len(selected_train_files)} samples from {candidate_number} samples')
        return selected_train_files, weight

    def update_dataloader_with_weights(self, candidate_loader,target_loader,selected_index, weight):
        candidate_num = len(candidate_loader.dataset)
        target_num = len(target_loader.dataset)

        def replace_dataloader_dataset_auto(original_loader, new_dataset):
            loader_params = vars(original_loader).copy()
            loader_params['dataset'] = new_dataset
            unwanted_keys = [key for key in loader_params.keys() if key.startswith('_')]
            for key in unwanted_keys:
                loader_params.pop(key, None)

            loader_params.pop("batch_sampler", None)
            
            updated_loader = DataLoader(**loader_params)

            return updated_loader
        def scale_weights_with_max_k(weights, N, M):
            k = weights.max()
            print(f'totoal number of weights: {len(weights)}')

            additional_weights = np.full(M, k)
            extended_weights = np.concatenate((weights, additional_weights))
            scaled_weights = extended_weights / extended_weights.sum() * N
            integer_weights = np.floor(scaled_weights).astype(int)
            difference = N - integer_weights.sum()
            fractional_parts = scaled_weights - integer_weights
            indices = np.argsort(-fractional_parts)
            for i in range(difference):
                integer_weights[indices[i]] += 1

            return integer_weights

        if self.data_weighting:
            weight = scale_weights_with_max_k(weight, candidate_num, target_num)
        else:
            weight = np.ones(candidate_num+target_num, dtype=int)
        if self.merged:
            candidate_dataset = candidate_loader.dataset
            target_dataset = target_loader.dataset
            merged_dataset = ConcatDataset([candidate_dataset, target_dataset])
            target_data_index = np.arange(candidate_num, candidate_num + target_num)
            select_and_target_index = np.concatenate([selected_index, target_data_index])

            repeated_indices = []
            for idx, weight in zip(select_and_target_index, weight):
                repeated_indices.extend([idx] * int(weight))
            subset_dataset = CustomSubset(merged_dataset, repeated_indices)

        else:
            weight_candidate = weight[:-target_num]
            repeated_indices = []
            for idx, weight in zip(selected_index, weight_candidate):
                repeated_indices.extend([idx] * int(weight))
            original_dataset = candidate_loader.dataset
            subset_dataset = CustomSubset(original_dataset, repeated_indices)
        print(len(subset_dataset))
        candidate_loader = replace_dataloader_dataset_auto(candidate_loader, subset_dataset)

        return candidate_loader




class CustomSubset(Dataset):
    def __init__(self, dataset, indices):

        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):

        original_idx = self.indices[idx]
        return self.dataset[original_idx]

    def __len__(self):

        return len(self.indices)