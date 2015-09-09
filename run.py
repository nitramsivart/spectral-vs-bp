#!/usr/bin/python3

from networkx import Graph, write_edgelist, write_weighted_edgelist
from custom_networkx import *
import os
from fast_sbm import fast_sbm
import scipy.sparse.linalg as lin
import scipy.sparse as sp
import numpy as np
import matplotlib.pylab as plt
import cProfile
from scipy.cluster.vq import whiten, kmeans, vq
from optparse import OptionParser
from multiprocessing import Process

def make_net(c_in, c_out, c_in2, n):
    return fast_sbm(c_in, c_out, c_in2, n)

def hashi_eig(A, degrees):
    I = sp.identity(A.shape[0])
    I_minus_D = sp.lil_matrix(A.shape)
    deg_list = list(degrees)
    for node,deg in deg_list:
        I_minus_D[node,node] = 1.0-deg
    crazy = sp.bmat([[None,I],[I_minus_D,A]])
    eig = lin.eigs(crazy, k=2, which="LR")[1][:len(deg_list),:2]
    #eig = lin.eigs(A, k=2, which="LR")[1][:,:2]
    return eig

def evaluate_vec(vec, n):
    score = 0
    median = np.median(vec)
    for i, val in enumerate(vec):
        if (i < n/2) == (val < median):
            score += 1./n

    return(max(score, 1-score))

def evaluate_eigs(eig0, eig1, n):
    print('start eig')
    score = 0
    vecs = np.array([eig0, eig1])
    whitened = whiten(vecs)
    centroids, _ = kmeans(whitened, 2, iter=2)
    idx, _ = vq(whitened, centroids)
    for i, val in enumerate(idx):
        if (i < n/2) == (val == 0):
            score += 1./n
    print('end eig')
    return(max(score, 1-score))

def evaluate_file(f, n):
    vec = []
    for l in open(f):
        vec.append(float(l.split()[0]))
    return(evaluate_vec(vec, n))

def run_bp(G_file, out_file):
    m_global = len(open(G_file).readlines())
    C_global = m_global / (n * (n-1.) / 2 - m_global)
    bp_file = 'cpp/bp'
    trials = 2
    os.system('%s %s %d %d %d %s %f %f' % (bp_file, G_file, 2, 2, trials, out_file, C_global, m_global))

def print_stats(heig1, heig2, eig1, eig2, degrees, stats_file):
    stats_file = open(stats_file, 'w+')
    for u, v, x, y, z in zip(heig1, heig2, eig1, eig2, degrees):
        stats_file.write('%f %f %f %f %d\n' % (u, v, x, y, z))

def read_stats(stats_file):
    try:
        stats_file = open(stats_file)
    except Exception:
        return [None]*5

    results = [[],[],[],[],[]]
    for line in stats_file:
        for i, x in enumerate(line.split()):
            results[i].append(float(x))
    return results
        

def calc_stats(c_cc, c_cp, x, n, i):
    for c_pp in x:
        print(c_pp)
        stats_file = 'data/stats/%.2f-%.2f-%.2f-%d-%d.stats' % (c_cc, c_cp, c_pp, n, i)
        G_file = 'data/graphs/%.2f-%.2f-%.2f-%d-%d.edges' % (c_cc, c_cp, c_pp, n, i)
        out_file = 'data/comms/%.2f-%.2f-%.2f-%d-%d.comms' % (c_cc, c_cp, c_pp, n, i)

        print('start make net')
        G = make_net(c_cc, c_cp, c_pp, n)
        nx.write_edgelist(G, G_file)

        A = to_scipy_sparse_matrix(G, dtype='d')
        try:
            #print('start heig')
            #heig = hashi_eig(A, G.degree_iter())
            eig = lin.eigsh(A, k=2, which="LA")[1][:,:2]
            heig = eig
            degrees = [d for _,d in G.degree_iter()]
            #run_bp(G_file, out_file)
            print(stats_file)
            #print_stats(eig0, eig1, eig0, eig1, degrees, stats_file)
            print_stats(heig[:,0], heig[:,1], eig[:,1], eig[:,0], degrees, stats_file)
        except Exception as e:
            print(e)
            pass


def make_plot(c_cc, c_cp, x, n, num_y=7, num_iters=2):
    # y contains prediction performance for a variety of measures, for varying c_pp
    y = []
    for i in range(num_y):
        array_of_performances = []
        for j in range(len(x)):
            array_of_performances.append([])
        y.append(array_of_performances)

    for i, c_pp in enumerate(x):
        print(c_pp)
        for j in range(num_iters):
            stats_file = 'data/stats/%.2f-%.2f-%.2f-%d-%d.stats' % (c_cc, c_cp, c_pp, n, j)
            out_file = 'data/comms/%.2f-%.2f-%.2f-%d-%d.comms' % (c_cc, c_cp, c_pp, n, j)
            heig0, heig1, eig0, eig1, degrees = read_stats(stats_file)
            if heig0 is None:
                print('error reading stats')
                continue
            else:
                print('read_stats')

            y[0][i].append(evaluate_vec(eig0, n))
            y[1][i].append(evaluate_vec(eig1, n))
            #y[2][i].append(evaluate_file(out_file, n))
            #y[3][i].append(evaluate_eigs(eig0, eig1, n))
            y[4][i].append(evaluate_vec(degrees, n))
            y[5][i].append(evaluate_vec(heig0, n))
            y[6][i].append(evaluate_vec(heig1, n))

    print('plotting')

    plt.xlabel('c_pp')
    plt.ylabel('fraction correctly identified')
    plt.title('c_cc = %d, c_cp = %d, n = %d' % (c_cc, c_cp, n))

    def get_means(list_of_lists):
        return [np.mean(l) for l in list_of_lists]
    plt.plot(x, get_means(y[0]),'ro',label='1st evec')
    plt.plot(x, get_means(y[1]),'bo',label='2nd evec')
    #plt.plot(x, get_means(y[5]),'r+',label='1st nb-evec')
    #plt.plot(x, get_means(y[6]),'b+',label='2nd nb-evec')
    #plt.plot(x, get_means(y[2]),'go',label='bp')
    #plt.plot(x, get_means(y[3]),'ks',label='kmeans w 2 evecs')
    plt.plot(x, get_means(y[4]),'ys',label='degree')
    plt.legend(loc=3)
    plt.savefig('plots/%.2f-%.2f-%d.pdf' % (c_cc, c_cp, n), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('--cc', type=float, dest = 'cc', help='core-core average degree', default=10)
    parser.add_option('--cp', type=float, dest = 'cp', help='core-periph average degree', default=6)
    parser.add_option('--pp_start', type=float, dest = 'pp_start', help='first periph-periph average degree', default=1)
    parser.add_option('--pp_end', type=float, dest = 'pp_end', help='last periph-periph average degree', default=6)
    parser.add_option('--pp_inc', type=float, dest = 'pp_inc', help='periph-periph average degree increment', default=.25)
    parser.add_option('-n', type=int, default=10000)
    parser.add_option('--run', action='store_true', default = False)
    parser.add_option('--plot', action='store_true', default = False)
    parser.add_option('--iters', type=int, help='number of instances to run, multithreaded', default = 2)

    (options, _) = parser.parse_args()
    c_cc = options.cc
    c_cp = options.cp
    n = options.n
    x = np.arange(options.pp_start, options.pp_end, options.pp_inc)

    if options.run:
        ps = []
        for i in range(options.iters):
            p = Process(target=calc_stats, args=(c_cc, c_cp, x, n, i))
            ps.append(p)
            p.start()
        for p in ps:
            p.join()
    if options.plot:
        make_plot(c_cc, c_cp, x, n, num_iters = options.iters)
