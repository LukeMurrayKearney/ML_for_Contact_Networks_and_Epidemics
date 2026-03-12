import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import csv
import sys
import json
import sklearn.mixture
import math

def main():
    # if len(sys.argv) != 3:
    #     print("Usage: python3 my_script.py <string1> <string2>")
    #     sys.exit(1)

    # data = sys.argv[1]
    # model = sys.argv[2]

    datas = ['comixa','comixb','comix3', 'poly']
    models = ['gmm']
    
    for data in datas:
        if data != 'poly':
            continue
        for model in models:
            n, iters = 100_000, 30

            buckets = np.array([])
            partitions_noage = [n]

            per_partition_noage = [a if i == 0 else a-partitions_noage[i-1] for i, a in enumerate(partitions_noage)]

            partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]
            per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]
            

            # datas = ['poly','comix1', 'comix2']
            # models = ['sbm', 'nbinom', 'dpln']

            # distance matrix of EMD
            bins = np.arange(0,len(partitions),1)
            distance_matrix = np.zeros((len(partitions), len(partitions)))
            for i in bins:
                for j in bins:
                    distance_matrix[i,j] = np.float64(np.abs(i-j))

            print(data, model)
            error, error_breakdown = [], []
            # error_with_itself, error_with_itself_breakdown = [], []
            # contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
            # params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
            
            with open(f'input_data/egos/{data}.json','r') as f:
                egos = json.load(f)
            with open(f'input_data/egos/{data}_noage.json', 'r') as f:
                egos_noage = json.load(f)
            contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
            
            with open(f'input_data/gmm/optimal_components_{data}_log_noage.json', 'r') as f:
                optimal_num_components = json.load(f)
            classifier = []
            for l, _ in enumerate(partitions_noage):
                classifier.append(sklearn.mixture.GaussianMixture(n_components=optimal_num_components[data][l], covariance_type='full'))
                egos_age = [a for a in egos_noage if a['age'] == l]
                ## use log(k+1) instead of k to fit
                X = [[math.log(b+1) for b in a['contacts']] for a in egos_age]
                classifier[l].fit(X)
            for i in range(iters):
                print(f'Iteration {i+1}/{iters}')
                samples_for_plot = []
                samples = []
                ## sample same number of people as the data
                for l, _ in enumerate(partitions_noage):
                    samples_for_plot.append([])
                    samples_tmp,_ = classifier[l].sample(per_partition_noage[l])
                    for sample in samples_tmp:
                        samples.append([int(np.round(np.exp(b)-1)) if int(np.round(np.exp(b)-1))>=0 else 0 for b in sample])
                        samples_for_plot[-1].append([int(np.round(np.exp(b)-1)) if int(np.round(np.exp(b)-1))>=0 else 0 for b in sample])
                params = None
                # my model error
                network = nd_p.build_network(n=n,partitions=partitions_noage,contact_matrix=contact_matrix,params=params,dist_type=model, degree_dist=samples)
                ages, cur = [], 0
                for age_group, part in enumerate(partitions):
                    for index in range(cur, int(part)):
                        ages.append(age_group)
                    cur = int(part)
                network['ages'] = ages
                network['partitions'] = [int(a) for a in partitions]
                
                freq_dist = np.zeros((len(ages), np.max(ages)+1))
                for index, links in enumerate(network['adjacency_matrix']):
                    for link in links:
                        freq_dist[index, network['ages'][network['ages'][link[1]]]] +=1
                network['frequency_distribution'] = freq_dist
                
                
                errors, err_pp = nd_p.emd_error(egos, network, distance_matrix=distance_matrix)
                error_breakdown.append(errors)
                error.append(err_pp)
                
                # network error of my model with true network
                # new_data = nd_p.data_from_network(network=network)
                # egos_itself, contact_matrix_itself, params_itself = nd_p.fit_to_data(df=new_data, save_fig=False, output_file_path="fits/network_comix1", buckets=buckets,dist_type=model)
                # network = nd_p.build_network(n=n, partitions=partitions, params=params_itself, contact_matrix=contact_matrix_itself,dist_type=model)
                
                # if i % 1 == 0:
                #     print(i)
                # errors, err_pp = nd_p.emd_error(egos=egos_itself, network=network, distance_matrix=distance_matrix)
                # error_with_itself_breakdown.append(errors)
                # error_with_itself.append(err_pp)
                
            with open(f'output_data/errors/breakdown_{data}_{model}_noage1.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for row in error_breakdown:
                    writer.writerow(row)
                    
            # with open(f'../output_data/errors/breakdown_itself_{data}_{model}.csv', 'w', newline='') as file:
            #     writer = csv.writer(file)
            #     for row in error_with_itself_breakdown:
            #         writer.writerow(row)

            with open(f'output_data/errors/{data}_{model}_noage1.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(error)


            # with open(f'../output_data/errors/itself_{data}_{model}.csv', 'w', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(error_with_itself)
            
            print(f'done: {data} {model}')
            
if __name__ == "__main__":
    main()