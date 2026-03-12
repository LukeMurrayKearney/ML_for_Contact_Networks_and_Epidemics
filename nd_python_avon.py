import nd_rust as nd_r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sc
import math
import itertools
from multiprocessing import Pool
import random
# import pyemd
import json
import sklearn.mixture
from scipy.optimize import minimize

################################## build into a package ##################################
def small_dur_gillesp(degree_dist, partitions, num_dur=5, tau=1, gamma=1/4, sigma=1, num_infec=1, props=[]):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.small_gillespie_dur(degree_dist, tau, partitions, outbreak_params, num_infec, num_dur, props)

def small_gillesp(degree_dist, partitions, tau=1, gamma=1/4, sigma=1, num_infec=1):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.small_gillespie(degree_dist, tau, partitions, outbreak_params, num_infec)

def small_dur_sbm(contact_matrix, partitions, num_dur=5, tau=1, gamma=1/4, sigma=1, num_infec=1, props=[]):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.small_sbm_dur(contact_matrix, tau, partitions, outbreak_params, num_infec, num_dur, props)

def gmm_gillesp(degree_dist, partitions, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillespie_gmm(degree_dist, taus, iterations, partitions, outbreak_params, num_infec)

def gmm_gillesp_sc(degree_dist, partitions, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillesp_gmm_sc(degree_dist, taus, iterations, partitions, outbreak_params, num_infec)

def sbm_gillesp(contact_matrix, partitions, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillespie_sbm(contact_matrix, taus, iterations, partitions, outbreak_params, num_infec)

def sbm_gillesp_dur(contact_matrix, partitions, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1, num_dur=5, props=[]):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillesp_dursbm_sc(contact_matrix, taus, iterations, partitions, outbreak_params, num_infec, num_dur, props)

def sbm_gillesp_dur_sc(contact_matrix, partitions, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1, num_dur=3, props=[]):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillesp_sbmdur_sc(contact_matrix, taus, iterations, partitions, outbreak_params, num_infec, num_dur, props)

def sbm_gillesp_sc(contact_matrix, partitions, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillesp_sbm_sc(contact_matrix, taus, iterations, partitions, outbreak_params, num_infec)

def sbm_gillesp_gr(contact_matrix, partitions, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillesp_sbm_gr(contact_matrix, taus, iterations, partitions, outbreak_params, num_infec)

def gmm_dur_gillesp(degree_dist, partitions, num_dur=5, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1, props=[]):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillesp_dur(degree_dist, taus, iterations, partitions, outbreak_params, num_infec, num_dur, props)

def gmm_dur_gillesp_sc(degree_dist, partitions, num_dur=5, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1, props=[]):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillesp_dur_sc(degree_dist, taus, iterations, partitions, outbreak_params, num_infec, num_dur, props)

def gmm_dur_gillesp_gr(degree_dist, partitions, num_dur=5, taus=np.arange(0.1,1,0.1), iterations=10, gamma=1/4, sigma=1, num_infec=1, props=[]):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0, sigma, gamma]
    return nd_r.gillesp_dur_gr(degree_dist, taus, iterations, partitions, outbreak_params, num_infec, num_dur, props)

def gmm_dur_network(degree_dist, partitions, num_dur=5, props=[]):
    partitions = [int(a) for a in partitions]
    return nd_r.network_dur(degree_dist, partitions, num_dur, props)

def gmm_dur_r0(degree_dist, partitions, num_dur=5, taus=np.arange(0.1,1,0.1), iterations=10, inv_gamma=7, prop_infec=1e-3, props=[]):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0,inv_gamma]
    return nd_r.dur_r0(degree_dist, taus, iterations, partitions, outbreak_params, prop_infec, num_dur, props)

def gmm_dur(degree_dist, partitions, num_dur=5, taus=np.arange(0.1,1,0.1), iterations=10, inv_gamma=7, prop_infec=1e-3, props=[]):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0,inv_gamma]
    return nd_r.sellke_dur(degree_dist, taus, iterations, partitions, outbreak_params, prop_infec, num_dur, props)

def build_network(n, partitions, contact_matrix, params=None, dist_type ="nbinom", degree_dist=None, num_dur=3, props=[1,0,0]):
    partitions = [int(a) for a in partitions]
    if dist_type == 'sbm':
        network = nd_r.sbm_from_vars(n, partitions, contact_matrix)
    elif dist_type == 'gmm':
        network = gmm_network(degree_dist, partitions)
    elif dist_type == 'gmm_dur':
        network = nd_r.network_dur(degree_dist, partitions, num_dur, props)
    elif dist_type == 'sbm_dur':
        network = nd_r.sbm_duration(n, partitions, contact_matrix, num_dur, props)
    else:
        if params is None:
            print("Parameters are required")
        network = nd_r.network_from_vars(n, partitions, dist_type, params, contact_matrix)
    return network

def gmm_network(degree_dist, partitions):
    partitions = [int(a) for a in partitions]
    return nd_r.network_from_source_and_targets(partitions, degree_dist)

def fit_to_data(df = None, input_file_path = 'input_data/poly.csv', dist_type = "sbm", buckets = np.array([5,12,18,30,40,50,60,70]), save_fig = True, output_file_path=None, log=False, to_csv=False, fig_data_file='',num_bins=15, duration=False):

    # Call the function with the provided arguments
    if df is None:
        df = read_in_dataframe(input_file_path)
    
    # Create list of ego networks from data
    egos = make_egos_list(df=df, buckets=buckets, duration=duration)
    
    # Create Contact Matrices
    contact_matrix, num_per_bucket = make_contact_matrices(egos=egos, duration=duration)
    
    return egos, contact_matrix

def fit_data_duration(df=None, buckets = np.array([5,12,18,30,40,50,60,70]), input_file_path='input_data/comix3.csv'):
    if df is None:
        df = read_in_dataframe(input_file_path)
        df['duration_multi'] = df['duration_multi'].astype('float')
    
    # Create list of ego networks from data
    egos = make_egos_list(df=df, buckets=buckets, duration=True)
    
    # Create Contact Matrices
    contact_matrix, num_per_bucket = make_contact_matrices(egos=egos, duration=True)
    
    return egos, contact_matrix


def to_networkx(network={}):
    G = nx.Graph()
    G.add_nodes_from(range(len(network['ages'])))
    nx.set_node_attributes(G,network['ages'], 'age')
    for person in network['adjacency_matrix']:
        for link in person:
            G.add_edge(link[0], link[1])
    return G    

################################################# utils ########################################################

def sample_egos_gmm(egos, partitions=[100], optimal_num_components=[1]):
    classifier = []
    samples = []
    per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]
    for l, _ in enumerate(partitions):
        classifier.append(sklearn.mixture.GaussianMixture(n_components=optimal_num_components[l], covariance_type='full'))
        egos_age = [a for a in egos if a['age'] == l]
        ## use log(k+1) instead of k to fit
        X = [[math.log(b+1) for b in a['contacts']] for a in egos_age]
        classifier[l].fit(X)
        ## sample same number of people as the data
        samples_tmp,_ = classifier[l].sample(per_partition[l])
        for sample in samples_tmp:
            samples.append([int(np.round(np.exp(b)-1)) if int(np.round(np.exp(b)-1))>=0 else 0 for b in sample])
    return samples

def read_in_dataframe(file_path):
    df = pd.read_csv(file_path)
    # remove NaNs
    df = df[df['part_age'].notna()]
    # remove values where ages not given
    rows_to_remove = df[(df['cnt_age_exact'].isna()) & (df['cont_id'].notna())].index
    df.drop(index=rows_to_remove, inplace=True)
    sorted_df = df.sort_values(by='part_id', ascending=False)
    return sorted_df

# Function to determine the bucket index for a given number
def get_bucket_index(num, buckets):
    for i, max_value in enumerate(buckets):
        if num < max_value:
            return i
    return len(buckets)  # If the number exceeds the last bucket, put it in the last bucket

def make_contact_matrices(egos,duration=False):
    num_per_bucket = np.zeros(np.max([a['age'] for a in egos])+1)
    contact_matrix = np.zeros((np.max([a['age'] for a in egos])+1, len(egos[0]['contacts'])))
    for ego in egos:
        num_per_bucket[ego['age']] += 1
        for j, val in enumerate(ego['contacts']):
            contact_matrix[ego['age'], j] += val
    contact_matrix = np.divide(contact_matrix.T, num_per_bucket).T
    return contact_matrix, num_per_bucket

# def make_contact_matrices(df, buckets):
#     num_per_bucket = np.zeros(len(buckets)+1)
#     contact_matrix = np.zeros((len(buckets)+1, len(buckets)+1))
#     # save last participant id
#     last_id = ''
#     # Iterate through the DataFrame and update the count_matrix
#     for _, row in df.iterrows():
#         b_i = get_bucket_index(row['part_age'], buckets=buckets)
#         if last_id != row['part_id']:
#             # count new participants
#             num_per_bucket[b_i] += 1
#         if pd.isnull(row['cont_id']) or pd.isnull(row['cnt_age_exact']):
#             continue
#         b_j = get_bucket_index(row['cnt_age_exact'], buckets=buckets)
#         contact_matrix[b_i, b_j] += 1
#         contact_matrix[b_j, b_i] += 1
#         last_id = row['part_id']
#     contact_matrix = np.divide(contact_matrix.T, num_per_bucket).T
#     return contact_matrix, num_per_bucket

def make_egos_list(df, buckets, duration=False):
    egos = []
    last = ''
    num_dur = int(np.max(df['duration_multi'][~np.isnan(df['duration_multi'])])) if duration==True else 1
    # iterate through each contact
    for _, x in df.iterrows():
        if x['part_id'] == last:
            if np.isnan(x['cnt_age_exact']):
                continue
            else:
                j = get_bucket_index(x['cnt_age_exact'], buckets=buckets)
                k = x['duration_multi']
                k = 1 if np.isnan(k) else int(k)
                egos[-1]['contacts'][j*num_dur + k-1] += 1
        else:
            i = get_bucket_index(x['part_age'], buckets=buckets)
            k = x['duration_multi']
            k = 1 if np.isnan(k) else int(k)
            egos.append({'age': i, 'contacts': np.zeros((len(buckets) + 1)*num_dur), 'degree': 0})
            if np.isnan(x['cnt_age_exact']):
                continue
            else:
                j = get_bucket_index(x['cnt_age_exact'], buckets=buckets)
                egos[-1]['contacts'][j*num_dur + k-1] += 1
        last = x['part_id']

    # count degree of each node
    for i, _ in enumerate(egos):
        egos[i]['degree'] = np.sum(egos[i]['contacts'])
    # sort the egos by age group
    egos = sorted(egos, key=lambda x: x['age'])
    
    return egos

      
def log_bins(x, num_bins=5):
    """
    Returns log bins of contacts, A^m
    Input: Contacts -> np array, num_bins -> int
    Output: Geometric center of bins -> ndarray, values in bins -> ndarray
    """
    # count_zeros = np.sum(x[x==0])
    count_zeros = len([a for a in x if a==0])
    x = np.sort([a for a in x if a > 0])
    max1, min1 = np.log(np.ceil(max(x))), np.log(np.floor(min(x)))
    x = np.log(x)
    t, freq, ends = np.zeros(num_bins), np.zeros(num_bins), np.zeros((2,num_bins))
    step = (max1 - min1)/num_bins
    for val in x:
        for k in range(num_bins):
            if k*step + min1 <= val and val < (k+1)*step + min1:
                freq[k] += 1
            t[k] = (k+1)*step - (.5*step) + min1
            ends[0,k] = k*step + min1
            ends[1,k] = (k+1)*step + min1
    freq[0] += count_zeros
    ends = np.exp(ends)
    widths = ends[1] - ends[0]
    freq = freq/widths/(len(x)+count_zeros)
    # freq = 1/np.sqrt(freq)*freq
    midpoints = np.exp(t)
    return midpoints, freq
    
# def calc_error(egos, network, distance_matrix, extra_mass_penalty, num_per_bucket):
    
#     # Begin parallelisation
#     # Number of threads
#     n = len(num_per_bucket)
#     # check that network is big enough 
#     for i in range(n):
#         if num_per_bucket[i] > network['partitions'][i]:
#             print("The network is smaller than the data.")
#             return None
    
#     # Data for each thread
#     data = [[] for _ in range(n)]
#     for ego in egos:
#         data[ego['age']].append(np.array(ego['contacts'], dtype=np.float64))
        
#     # network egos for each thread, big nasty line but it just samples randomly from the network to match age distribution of data
#     indices = [random.sample(range(0, network['partitions'][i]), num_per_bucket[i]) if i == 0 else random.sample(range(network['partitions'][i-1], network['partitions'][i]), num_per_bucket[i]) for i in range(n)]
#     network_data = [[np.array(network['frequency_distribution'][idx], dtype=np.float64) for idx in age_class] for age_class in indices]
    
#     # run calculation of the W matrix then pool and collect results
#     with Pool(n) as pool:
#         combined_args = zip(data, network_data, [distance_matrix for _ in range(n)], [extra_mass_penalty for _ in range(n)])

#         # map() preserves the order of results
#         W = pool.map(calc_error_bucket, combined_args)
        
#     # run matching problem in parallel and then pool and collect the results
#     with Pool(n) as pool:
#         total_errors = pool.map(solve_matching_problem, W)
        
#     # calculate total error per person
#     errors_per_person = np.sum(total_errors) / np.sum(num_per_bucket)
#     #normalise the error to be errors per person
#     error = np.divide(total_errors, num_per_bucket)
    
#     return error, errors_per_person

# def calc_error_bucket(arguments):
    
#     # Calculating the matrix of Wasserstein distances for each pair in an age group
#     data, network_data, distance_matrix, extra_mass_penalty = arguments
#     W_a = np.zeros((len(data), len(data)))
#     for i, ego in enumerate(data):
#         for j, ego_node in enumerate(network_data):
#             # extra mass penalty = A/(mean_mass + 1)
#             penalty = extra_mass_penalty / ((np.sum(ego) + np.sum(ego_node))/2 + 1)
#             W_a[i,j] = pyemd.emd(ego, ego_node, distance_matrix, extra_mass_penalty=penalty)
#     return W_a

# def solve_matching_problem(W_a):
    
#     # find the optimal matching of the W matrix
#     row_ind, col_ind = sc.optimize.linear_sum_assignment(W_a)
#     return W_a[row_ind, col_ind].sum()
