import os
import sys
import numpy as np
import cvxpy as cp
import json
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torch
import datetime
import time
from scipy.special import comb
import concurrent.futures
import random
import pandas as pd
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import imagenet_utils
from structures import imagenet
from utils import coverage_plots, pred_set_size_plots

# ======= User-Defined Parameters =======
# This section allows users to set custom parameters for the experiment.
# This section allows users to set custom parameters for the experiment.
SEEDS = [42]
CALIBRATION_RATIO = 0.2
MAX_WORKERS = 500

# Number of samples
N = 1000

# Coverage guarantee
GUARANTEE = "marginal"   # "marginal" or "pac"

# Hyperparameters
EPSILON_VALUES = [0.05, 0.1]
PAC_DELTA_VALUES = [0.1, 0.01, 0.001]
M_VALUES = [4,8]

# Threshold candidates
THRESHOLDS = [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]

# =======================================



def add_leaves_prob(prob, hierarchical_tree, category_names):
    leaves = hierarchical_tree.get_leaves()
    for leaf in leaves:
        leaf.prob.value = prob[category_names.index(leaf.label)]
        

def generate_pred_sets(probs, tau, tauParam, hierarchical_tree, category_names, problem, warm_start_values):
    solve_times = [] # time for solving each integer programming problem
    tauParam.value = tau
    pred_sets = []
    for j, prob in enumerate(probs):
        add_leaves_prob(prob, hierarchical_tree, category_names)
        if warm_start_values is not None:
            for node in hierarchical_tree.get_nodes():
                if node in warm_start_values:
                    node.alpha.value = int(warm_start_values[node].get('alpha', None))
                    node.beta.value = int(warm_start_values[node].get('beta', None))

        start_time = time.time()
        #problem.solve(solver=cp.GUROBI, warm_start=True)
        #problem.solve(solver=cp.ECOS, warm_start=True)
        problem.solve(solver=cp.GLPK_MI)
        end_time = time.time()
        solve_time = end_time - start_time  
        solve_times.append(solve_time)

        pred_set = [node.label for node in hierarchical_tree.get_nodes() if node.alpha.value >= 0.5]
        pred_sets.append(pred_set)

        warm_start_values = {
            node: {'alpha': node.alpha.value, 'beta': node.beta.value} for node in hierarchical_tree.get_nodes()
        }
        print('beta value',  warm_start_values['beta'])

    total_elements = sum(len(sublist) for sublist in pred_sets)
    print("Total elements:", total_elements)
    for i, sublist in enumerate(pred_sets):
        print(f"Prediction {i}: {len(sublist)} elements")
    print('length pred_sets', len(pred_sets))
    
    
    return pred_sets, sum(solve_times) / len(solve_times)



def compute_coverage_and_avg_size(pred_sets, labels, tree_nodes_dict, category_names):
    coverage = []
    sizes = []
    for i, pred_set in enumerate(pred_sets):
        label_name = category_names[labels[i]]
        leaves_name = []
        for pred in pred_set:
            for leaf in tree_nodes_dict[pred].get_leaves():
                leaves_name.append(leaf.label)
        
        sizes.append(len(leaves_name))
        coverage.append(1 if label_name in leaves_name else 0)
    
    return coverage, sizes



def evaluate_model(probs, labels):
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == labels)
    print(f"ImageNet accuracy: {accuracy}%")
    return accuracy



def read_worker(results_path, qualitative_samples, hierarchy, category_names, test_probs, test_labels, epsilon, delta, m):
    df = pd.read_csv(results_path)

    filtered_df = df[(df['seed'] == 2) & 
                     (df['epsilon'] == epsilon) & 
                     (df['delta'] == delta) & 
                     (df['m'] == m)]
    
    if filtered_df.shape[0] != 1:
        raise ValueError("Filtered DataFrame does not contain exactly one row. Check input parameters or CSV data.")
    
    threshold = filtered_df['threshold'].values[0]

    # Construct hierarchical tree
    hierarchical_tree = imagenet.construct_imagenet_tree(hierarchy)
    tree_nodes_dict = {}
    for n in hierarchical_tree.get_nodes():
        tree_nodes_dict[n.label] = n
    warm_start_values = None 

    # Build constraint problem
    mParam = cp.Parameter(integer=True)
    tauParam = cp.Parameter(nonneg=True)
    problem = hierarchical_tree.build_constraint_problem(mParam, tauParam)
    mParam.value = m

    test_pred_sets, _ = generate_pred_sets(test_probs, threshold, tauParam, hierarchical_tree, category_names, problem, warm_start_values)
    for i in range(len(qualitative_samples)):
        if GUARANTEE == "marginal":
            key = f"e={epsilon},m={m}"
        elif GUARANTEE == "pac":
            key = f"e={epsilon},m={m},d={delta}"
        if qualitative_samples[i]['label'] == category_names[test_labels[i]]:
            qualitative_samples[i][key] = test_pred_sets[i]
        else:
            raise Exception("Encountered mismatched label name.")
        


def compute_worker(hierarchy, category_names, cal_probs, cal_labels, test_probs, test_labels, epsilon, delta, m, seed):
    print(f"({seed}) Starting experiment for  m={m}, epsilon={epsilon}, delta={delta} ...")
    # Construct hierarchical tree
    hierarchical_tree = imagenet.construct_imagenet_tree(hierarchy)
    tree_nodes_dict = {}
    for n in hierarchical_tree.get_nodes():
        tree_nodes_dict[n.label] = n
    warm_start_values = None 

    # Build constraint problem
    mParam = cp.Parameter(integer=True)
    tauParam = cp.Parameter(nonneg=True)
    problem = hierarchical_tree.build_constraint_problem(mParam, tauParam)
    mParam.value = m

    # Compute threshold given input m, epsilon, (and delta) on calibration data
    n_cal = len(cal_probs)
    threshold_idx = -1
    cal_coverage = None
    cal_avg_size = None
    phi = np.zeros(len(THRESHOLDS))

    start_time = time.time()
    for i, tau in enumerate(THRESHOLDS):
        cal_pred_sets, _ = generate_pred_sets(cal_probs, tau, tauParam, hierarchical_tree, category_names, problem, warm_start_values)
        coverage, size = compute_coverage_and_avg_size(cal_pred_sets, cal_labels, tree_nodes_dict, category_names)
        
        def compute_h(n, epsilon, delta):
            cumulative_sum = 0  
            h_hat = 0
            for h in range(n + 1):  
                binomial_term = comb(n, h) * (epsilon ** h) * ((1 - epsilon) ** (n - h))
                cumulative_sum += binomial_term
                if cumulative_sum >= delta:
                    break  
                h_hat = h
            return h_hat
        
        if GUARANTEE == 'marginal':
            phi[i] = (n_cal - sum(coverage)) <= (epsilon * (n_cal + 1))
        elif GUARANTEE == 'pac':
            phi[i] = (n_cal - sum(coverage)) <= compute_h(n_cal, epsilon, delta)

        # Validate tau
        if np.sum(phi[:i+1] == 0) <= 0:
            threshold_idx = i
            cal_coverage = coverage
            cal_avg_size = size
        else:
            end_time = time.time()
            threshold_avg_estimate_time = end_time - start_time  
            if i == 0:
                raise Exception("No valid threshold candidate found. Try using larger threshold values.")
            threshold = THRESHOLDS[threshold_idx]
            print(f"({seed}) Found threshold={threshold} given m={m}, epsilon={epsilon}, delta={delta} ...")

            # Compute coverage rate and average size for testing data
            test_pred_sets, avg_problem_solve_time = generate_pred_sets(test_probs, threshold, tauParam, hierarchical_tree, category_names, problem, warm_start_values)
            test_coverage, test_avg_size = compute_coverage_and_avg_size(test_pred_sets, test_labels, tree_nodes_dict, category_names)
            print(f"with test coverage: {np.mean(test_coverage)}, test avg_size: {np.mean(test_avg_size)}.")
            
            cost_function_test_Set = len(test_pred_sets) + np.mean(test_coverage) * np.mean(test_avg_size)
            print('test prediction length', len(test_pred_sets))
            cost_function_Cal_Set = len(cal_pred_sets) + np.mean(cal_coverage) * np.mean(cal_avg_size)
            print('cal prediction length', len(cal_pred_sets) )

            result = [seed, epsilon, delta, m, threshold, np.mean(cal_coverage), np.mean(cal_avg_size), np.mean(test_coverage), np.mean(test_avg_size), threshold_avg_estimate_time, avg_problem_solve_time, cost_function_test_Set, cost_function_Cal_Set]
            detailed_result = []
            for test_c, test_s in zip(test_coverage, test_avg_size):
                detailed_result.append([seed, epsilon, delta, m, threshold, test_c, test_s, np.mean(test_coverage), np.mean(test_avg_size)])
            return result, detailed_result



def guarantee_tests(hierarchy, category_names, cal_probs, cal_labels, test_probs, test_labels, seed):
    if GUARANTEE == "pac":
        delta_values = PAC_DELTA_VALUES
    else:
        delta_values = [-1]

    total_tasks = len(EPSILON_VALUES) * len(delta_values) * len(M_VALUES)
    completed_tasks = 0

    results = []
    detailed_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_params = {
            executor.submit(
                compute_worker, hierarchy, category_names,
                cal_probs, cal_labels, test_probs, test_labels,
                epsilon, delta, m, seed
            ): (epsilon, delta, m)
            for epsilon in EPSILON_VALUES
            for delta in delta_values
            for m in M_VALUES
        }

        for future in concurrent.futures.as_completed(future_to_params):
            print("future in concurrent.futures:", future)
            epsilon, delta, m = future_to_params[future]
            
            try:
                res = future.result()

                if res is None or not isinstance(res, tuple) or len(res) != 2:
                    print(f"⚠️ Skipping task due to invalid result (None or wrong format): m={m}, epsilon={epsilon}, delta={delta}")
                    continue

                #print(f"Returned from worker: {res}")  # Debug
                result, detailed_result = res  # Use the value already fetched
                results.append(result)
                detailed_results.extend(detailed_result)
                completed_tasks += 1

                print(f"({seed}) Completed task {completed_tasks}/{total_tasks}: m={m}, epsilon={epsilon}, delta={delta}.")

            except Exception as e:
                print(f"❌ Exception occurred for m={m}, epsilon={epsilon}, delta={delta}: {e}")
                continue

    return results, detailed_results



def save_qualitative_examples(qual_results_path, results_path, hierarchy, category_names, test_probs, test_labels, image_indices):
    if GUARANTEE == "pac":
        delta_values = PAC_DELTA_VALUES
    else:
        delta_values = [-1]

    total_tasks = len(EPSILON_VALUES) * len(delta_values) * len(M_VALUES)
    completed_tasks = 0
    
    qualitative_samples = []
    for i, test_label in enumerate(test_labels):
        qualitative_sample = {}
        qualitative_sample["label"] = category_names[test_label]
        qualitative_sample["image_idx"] = image_indices[i]
        qualitative_samples.append(qualitative_sample)

    for epsilon in EPSILON_VALUES:
        for m in M_VALUES:
            for delta in delta_values:
                read_worker(results_path, qualitative_samples, hierarchy, category_names, test_probs, test_labels, epsilon, delta, m)
                completed_tasks += 1
                print(f"Completed {completed_tasks}/{total_tasks} tasks.")

    with open(qual_results_path, 'w') as f:
        json.dump(qualitative_samples, f, indent=4)
    print(f"Saved qualitative samples to {qual_results_path} successfully.")



def Compare_results(hcp_path, base_path ):
    # Define file paths

    hcp_df = pd.read_csv(hcp_path)
    base_df = pd.read_csv(base_path)
    final_columns = [
        'Base_model_threshold', 'HCP_threshold',
        'Base_model_cal_coverage', 'HCP_cal_coverage',
        'Base_model_cal_avg_size', 'HCP_cal_avg_size',
        'Base_model_test_coverage', 'HCP_test_coverage',
        'Base_model_test_avg_size', 'HCP_test_avg_size',
        'Base_model_time_solve_threshold', 'HCP_time_solve_threshold',
        'Base_model_avg_time_per_test', 'HCP_avg_time_per_test',
        'Base_model_cost_function_test_Set', 'HCP_cost_function_test_Set',
        'Base_model_cost_function_Cal_Set', 'HCP_cost_function_Cal_Set'
    ]
    # Check minimum number of columns to avoid indexing errors
    min_cols = min(len(hcp_df.columns), len(base_df .columns))

    # First 4 columns from data1
    merged_df = hcp_df.iloc[:, :4].copy()
    j=0
    # Now combine from column 5 onward as per your logic
    for i in range(4, min_cols):  # Start from 4 (5th column, 0-indexed)
        merged_df[final_columns[j] ] = base_df.iloc[:, i]
        j=j+1
        merged_df[final_columns[j]] = hcp_df.iloc[:, i]
        j=j+1

    return merged_df



def HCP_model (weights):
    if os.path.exists('../collected/imagenet/HCP_predictions.npy') and os.path.exists('../collected/imagenet/HCP_labels.npy'):
        with open('../collected/imagenet/HCP_predictions.npy', 'rb') as f:
            probs = np.load(f) # 50000 x 1000
        with open('../collected/imagenet/HCP_labels.npy', 'rb') as f:
            labels = np.load(f) # 50000
    else:
                # Load your CSV file
        df = pd.read_csv('../predictions_val_img.csv')  # replace with your actual file path

        # Extract labels_list: a list of numpy arrays, one per sample
        # (Optional batching is skipped; list of arrays with one item each)
        labels_list = [np.array([label]) for label in df['label'].values]
        # print("type labels_list:", labels_list)
        # Extract probs_list: a list of numpy arrays, one per sample
        # Each row has probabilities for 10 classes
        prob_cols = [f'label_{i}' for i in range(1000)]
        # print("type of prob_cols", prob_cols)
        probs_list = [row.values for _, row in df[prob_cols].iterrows()]
        #print("probs_list_shap", probs_list.shape)

        probs = np.array(probs_list)
        labels = np.concatenate(labels_list)

        with open('../collected/imagenet/HCP_predictions.npy', 'wb') as f:
            np.save(f, probs)
        with open('../collected/imagenet/HCP_labels.npy', 'wb') as f:
            np.save(f, labels)
        print(f"Saved ImageNet data.")
    return probs, labels #probs[:10000,:], labels[:10000]

def base_model(weights):
     
    if os.path.exists('../collected/imagenet/predictions.npy') and os.path.exists('../collected/imagenet/labels.npy'):
        with open('../collected/imagenet/predictions.npy', 'rb') as f:
            probs = np.load(f) # 25000 x 1000
        with open('../collected/imagenet/labels.npy', 'rb') as f:
            labels = np.load(f) # 25000
    else:
        # Step 1: Initialize model with the best available weights
        model = resnet50(weights=weights).to("cuda")
        model.eval()

        # Step 2: Initialize the inference transforms
        preprocess = weights.transforms()

        imagenet_data = torchvision.datasets.ImageNet('../data/imagenet', split='val', transform=preprocess)
        data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size=64,
                                                  shuffle=True)
        probs_list = []
        labels_list = []
        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
                # input images to the model
                pred = model(images.cuda())
                # compute softmax probabilities
                pred = pred.softmax(1)
                probs_list.append(pred.cpu().detach().numpy())
                labels_list.append(labels.numpy())

        probs = np.concatenate(probs_list)
        labels = np.concatenate(labels_list)

        with open('../collected/imagenet/predictions.npy', 'wb') as f:
            np.save(f, probs)
        with open('../collected/imagenet/labels.npy', 'wb') as f:
            np.save(f, labels)
        print(f"Saved ImageNet data.")
    return  probs, labels #probs[:1000,:], labels[:1000] # probs, labels

def run_experiment(probs, labels, results_dir, filename ):
     

    print(f"Loaded ImageNet data.")
    model_accuracy = evaluate_model(probs, labels)

    # Get all parent labels
    hierarchy = imagenet_utils.read_hierarchy(category_names)
    parents = imagenet_utils.find_all_parents(hierarchy)

    # Conduct guarantee tests
    seed_results = []
    seed_detailed_results = []
    for seed in SEEDS:
        print(f"Set seed={seed}.")
        random.seed(seed)
        indices = list(range(len(probs)))
        random.shuffle(indices)
        n_cal = int(CALIBRATION_RATIO * N)
        cal_probs = [probs[i] for i in indices[:n_cal]]
        cal_labels = [labels[i] for i in indices[:n_cal]]
        test_probs = [probs[i] for i in indices[n_cal:N]]
        test_labels = [labels[i] for i in indices[n_cal:N]]

        results, detailed_results = guarantee_tests(hierarchy, category_names, cal_probs, cal_labels, test_probs, test_labels, seed)
        seed_results.append(results)
        seed_detailed_results.append(detailed_results)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = results_dir
    results_filename = f"{filename}_{GUARANTEE}_n{N}_calib{CALIBRATION_RATIO}_{timestamp}.csv"
    results_path = os.path.join(results_dir, results_filename)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(results_path, 'w') as f:
        f.write("seed,epsilon,delta,m,threshold,cal_coverage,cal_avg_size,test_coverage,test_avg_size,time_solve_threshold,avg_time_per_test,  cost_function_test_Set, cost_function_Cal_Set\n")
        for seed_result in seed_results:
            for result in seed_result:

                f.write(','.join(map(str, result)) + '\n')
    print(f"Results saved to {results_filename} successfully.")

        # Save all detailed results
    detailed_results_dir = results_dir + "/details"
    #detailed_results_filename = f"HCP_{GUARANTEE}_n{N}_calib{CALIBRATION_RATIO}_details_{timestamp}.csv"
    detailed_results_filename = f"detailed_results_{filename}_{GUARANTEE}_n{N}_calib{CALIBRATION_RATIO}_details_{timestamp}.csv"
    detailed_results_path = os.path.join(detailed_results_dir, detailed_results_filename)
    if not os.path.exists(detailed_results_dir):
        os.makedirs(detailed_results_dir)
    with open(detailed_results_path, 'w') as f:
        f.write("seed,epsilon,delta,m,threshold,test_covered,test_size,test_coverage,test_avg_size, cost_function_test_Set, cost_function_Cal_Set, mean_cost_function_Cal_Set, mean_cost_function_test_Set\n")
        for seed_result in seed_detailed_results:
            for result in seed_result:
                f.write(','.join(map(str, result)) + '\n')
    print(f"Detailed results saved to {detailed_results_filename} successfully.")


if __name__ == '__main__':

        # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Compare HCP with Existing_model")
    parser.add_argument('--run_existing_model', action='store_true', help='run and produce result on HCP') # use store_false if you want run existing model
    args = parser.parse_args()

    weights = ResNet50_Weights.DEFAULT
    category_names = weights.meta["categories"]
    if args.run_existing_model:
        print("\n=== Running base model ===")

        probs, labels = base_model(weights)
            # Save all results
        results_dir = "../results/experiments/imagenet"
        #results_filename = f"HCP_{GUARANTEE}_n{N}_calib{CALIBRATION_RATIO}_{timestamp}.csv"
        results_filename = "base_model"
        save_results= run_experiment(probs, labels, results_dir, results_filename )
            
    else:
        print(f"\n=== Running HCP ===")
        probs, labels = HCP_model(weights)
                    # Save all results
        results_dir = "../results/experiments/imagenet"
        #results_filename = f"HCP_{GUARANTEE}_n{N}_calib{CALIBRATION_RATIO}_{timestamp}.csv"
        results_filename = "HCC"
        save_results= run_experiment(probs, labels, results_dir, results_filename )
 

        # provide the path of both the files to produce the comparison results:
        hcc_path = '../experiments/imagenet/HCP_marginal_n1000_calib0.2_20250731_222619.csv'
        base_model_path = '../results/experiments/imagenet/marginal_n1000_calib0.2_20250731_224722.csv'
        Comparison_results = Compare_results(hcc_path, base_model_path)
        # Save to CSV
        # Save comparison file
        # Define output directory and file
        output_dir = '../comparison_outputs'
        output_filename = 'Imagenet_comparison_results.csv'
        output_path = os.path.join(output_dir, output_filename)
        # Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        Comparison_results.to_csv(output_path, index=False)
        print(f"Comparison results saved to: {output_path}")
    