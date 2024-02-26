import numpy as np
import collections
import os
import sys
import pickle
import random
import csv
import pandas as pd
from collections import defaultdict
from collections import Counter 
import time
import multiprocessing
import itertools
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from functools import partial
import json


def load_data(args):
    n_genem, n_genen, train_data, eval_data, test_data, genem_history_dict, genen_set_most_pop = load_rating(args)
    entity_index_2_name, rela_index_2_name = {}, {}

    kg, n_entity, n_relation, adj_entity, adj_relation, genem_path, genem_path_top_k = load_kg(args, train_data)
    print('data loaded.')
    genem_triplet_set = get_genem_triplet_set(args, kg, genem_history_dict)
    all_genem_entity_count = get_all_genem_entity_count(args,train_data, kg, adj_entity, adj_relation, hop=args.h_hop)
    return n_genem, n_genen, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, genem_triplet_set, \
            genem_path, genem_path_top_k, genen_set_most_pop, genem_history_dict, entity_index_2_name, rela_index_2_name

def load_rating(args):

    print('reading sl file ...')

    rating_file = args.path.data + 'sl_file'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_genem = max(set(rating_np[:, 0])) + 1
    n_genen = max(set(rating_np[:, 1])) + 1

    top_k = 500

    if os.path.exists(f"{args.path.misc}KGNN_pop_genen_set_{top_k}.pickle") == False:
        genen_count = {}
        for i in range(rating_np.shape[0]):
            genen = rating_np[i, 1]
            if genen not in genen_count:
                genen_count[genen] = 0
            genen_count[genen] += 1
        genen_count = sorted(genen_count.items(), key=lambda x: x[1], reverse=True)
        genen_count = genen_count[:top_k]
        genen_set_most_pop = [genen_set[0] for genen_set in genen_count]
        with open(f"{args.path.misc}KGNN_pop_genen_set_{top_k}.pickle", 'wb') as fp:
            pickle.dump(genen_set_most_pop, fp)

    with open(f"{args.path.misc}KGNN_pop_genen_set_{top_k}.pickle", 'rb') as fp:
        genen_set_most_pop = pickle.load(fp)

    genen_set_most_pop = set(genen_set_most_pop)
    u_counter, i_counter = {}, {}
    if args.new_load_data == True:
        print('load new train eval test')
        train_data, eval_data, test_data = dataset_split(rating_np, args)
        with open(f"{args.path.data}train_data.pickle",'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{args.path.data}eval_data.pickle",'wb') as f:
            pickle.dump(eval_data, f)
        with open(f"{args.path.data}test_data.pickle",'wb') as f:
            pickle.dump(test_data, f)
    else:
        train_data, eval_data, test_data = load_pre_data(args)


    genem_history_dict = dict()
    for i in range(train_data.shape[0]):
        genem = train_data[i][0]
        genen = train_data[i][1]
        rating = train_data[i][2]
        if rating == 1:
            if genem not in genem_history_dict:
                genem_history_dict[genem] = []
            genem_history_dict[genem].append(genen)

    train_indices = [i for i in range(train_data.shape[0]) if train_data[i][0] in genem_history_dict]
    eval_indices = [i for i in range(eval_data.shape[0]) if eval_data[i][0] in genem_history_dict]
    test_indices = [i for i in range(test_data.shape[0]) if test_data[i][0] in genem_history_dict]

    train_data = train_data[train_indices]
    eval_data = eval_data[eval_indices]
    test_data = test_data[test_indices]

    return n_genem, n_genen, train_data, eval_data, test_data, genem_history_dict, genen_set_most_pop


def get_all_genem_entity_count(args,train_data, kg, adj_entity, adj_relation, hop=0):
    args.genem_neighbor_rate = [0,0,0]



def load_pre_data(args):
    train_data = pd.read_csv(f'{args.path.data}train.txt',index_col=None, sep=' ')
    train_data.columns = ["genem","genen","like"]
    train_data = train_data[['genem', 'genen', 'like']].values

    eval_data = pd.read_csv(f'{args.path.data}dev.txt', sep=' ')
    eval_data.columns = ["genem", "genen", "like"]
    eval_data = eval_data[['genem', 'genen', 'like']].values


    test_data = pd.read_csv(f'{args.path.data}test.txt',index_col=None,header=None, sep=' ')
    test_data.columns = ["genem","genen","like"]
    test_data = test_data[['genem', 'genen', 'like']].values

    return train_data, eval_data, test_data




def dataset_split(rating_np, args):
    print('splitting dataset ...')

    eval_ratio = 0.1
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))


    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]


    return train_data, eval_data, test_data


def load_kg(args, train_data):
    print('reading KG file ...')

    # reading kg file
    kg_file = args.path.data + 'kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    rating_file = args.path.data + 'sl_file'
    if os.path.exists(rating_file + '.npy'):
        ratings_final = np.load(rating_file + '.npy')
    else:
        ratings_final = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', ratings_final)

    genem_num = len(set(ratings_final[:,0]))
    genen_num = len(set(ratings_final[:,1]))

    average_genem_num = len(ratings_final) / genem_num
    arverage_genen_num = len(ratings_final) / genen_num

    kg, enti, rela = construct_kg(args,kg_np)


    adj_entity, adj_relation, genem_path = None, None, None

    adj_entity, adj_relation = construct_adj(args, kg, n_entity)
    genem_path_top_k = None

    return kg, n_entity, n_relation, adj_entity, adj_relation, genem_path, genem_path_top_k


def construct_kg(args,kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    enti = 0
    rela = 0
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

        enti = max(enti, head, tail)
        rela = max(rela, relation)

    return kg, enti, rela

def construct_genem_kg(kg, train_data, genen_num, n_relation):
    print('constructing genem knowledge graph ...')
    enti = 0
    rela = 0
    for i in range(train_data.shape[0]):
        genem = train_data[i][0]
        genen = train_data[i][1]
        rating = train_data[i][2]

        if rating == 1:
            head = genem + genen_num
            tail = genen
            if head not in kg:
                kg[head] = []
            kg[head].append((tail, n_relation))
            if tail not in kg:
                kg[tail] = []
            kg[tail].append((head, n_relation))

            enti = max(enti, head, tail)
            rela = max(rela, n_relation)
    return kg, enti, rela



def construct_adj(args, kg, entity_num, random_seed = 1):
    adj_entity, adj_relation = contruct_random_adj(args,kg,entity_num)
    return adj_entity, adj_relation


def contruct_random_adj(args,kg,entity_num):

    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        if entity in kg:
            neighbors = kg[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= args.neighbor_sample_size: sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
            else: sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation
                        


def get_genem_triplet_set(args, kg, genem_history_dict):
    print('constructing ripple set ...')
    # genem -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    genem_triplet_set = collections.defaultdict(list)
    # entity_interaction_dict = collections.defaultdict(list)
    global g_kg
    g_kg = kg
    with mp.Pool(processes=min(mp.cpu_count(), 12)) as pool:
        job = partial(_get_genem_triplet_set, p_hop=max(1,args.p_hop), n_memory=args.n_memory, n_neighbor=16)#888888888888888888888888888888888888888888888888888888888888
        for m, m_r_set, m_interaction_list in pool.starmap(job, genem_history_dict.items()):
            genem_triplet_set[m] = np.array(m_r_set, dtype=np.int32)
            # entity_interaction_dict[u] = u_interaction_list
    del g_kg
    return genem_triplet_set

def _get_genem_triplet_set(genem, history, p_hop=2, n_memory=32, n_neighbor=16):
    ret = []
    entity_interaction_list = []
    for h in range(max(1,p_hop)):#8888888888888888888888888888888888888888888888888888888888888888888888888888
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = history
        else:
            tails_of_last_hop = ret[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in random.sample(g_kg[entity], min(len(g_kg[entity]), n_neighbor)):
                memories_h.append(entity)
                memories_r.append(tail_and_relation[1])
                memories_t.append(tail_and_relation[0])


        if len(memories_h) == 0:
            ret.append(ret[-1])
        else:
            # sample a fixed-size 1-hop memory for each genem
            replace = len(memories_h) < n_memory
            indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            entity_interaction_list += zip(memories_h, memories_r, memories_t)
            ret.append([memories_h, memories_r, memories_t])
            
    return [genem, ret, list(set(entity_interaction_list))]