import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import csv
import sys
import json
import sklearn.mixture
import math

def main():
    datas = ['comixa','comixb', 'comix3', 'poly']
    model = ['ER']
    
    for data in datas:
        if data == 'comixa':
            continue
        print(data)
        n, iters = 100_000, 30

        partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]
        per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

        bins = np.arange(0,len(partitions),1)
        distance_matrix = np.zeros((len(partitions), len(partitions)))
        for i in bins:
            for j in bins:
                distance_matrix[i,j] = np.float64(np.abs(i-j))

        error, error_breakdown = [], []

        with open(f'input_data/egos/{data}.json','r') as f:
            egos = json.load(f)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        
        mean_degree = np.zeros(len(partitions))
        for person in egos:
            mean_degree[person['age']] += person['degree']
        mean_degree /= np.array(per_partition)
        print(mean_degree)
        mean_degree = np.sum(mean_degree)
        print(mean_degree)
        for i in range(iters):
            if i%2==0:
                print(f'  iteration {i}/{iters}')
            network = nd_p.build_ER_network(partitions=partitions, mean_degree=mean_degree)
            errors, err_pp = nd_p.emd_error(egos, network, distance_matrix=distance_matrix)
            error_breakdown.append(errors)
            error.append(err_pp)

            
        with open(f'output_data/errors/breakdown_{data}_{model}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in error_breakdown:
                writer.writerow(row)
                

        with open(f'output_data/errors/{data}_{model}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(error)

        print(f'done: {data} {model}')
        
if __name__ == "__main__":
    main()