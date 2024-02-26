import os
import sys

import tensorflow as tf

import numpy as np
from time import time
from model import MPASL
from train_util import Early_stop_info, Eval_score_info, Train_info_record_sw_emb
from metrics import ndcg_at_k, map_at_k, recall_at_k, hit_ratio_at_k, mrr_at_k, precision_at_k
import pickle


def topk_settings(args, show_topk, train_data, eval_data, test_data, n_genen, save_record_genem_list, save_genem_list_name):
    if show_topk or True:
        genem_num = 250
        k_list = [1, 2, 5, 10, 25, 50, 100]
        train_record = get_genem_record(train_data, True)
        test_record = get_genem_record(test_data, False)
        eval_record = get_genem_record(eval_data, False)

        if True or os.path.exists(args.path.misc + 'genem_list_' + save_genem_list_name + "_" + str(genem_num) + '.pickle') == False:
            genem_list = list(set(train_record.keys()) & (set(test_record.keys() & set(eval_record.keys()))))

            genem_counter_dict = {genem:len(train_record[genem]) for genem in genem_list}
            genem_counter_dict = sorted(genem_counter_dict.items(), key=lambda x: x[1], reverse=True)
            genem_counter_dict = genem_counter_dict[:genem_num]
            genem_list = [genem_set[0] for genem_set in genem_counter_dict]

            if len(genem_list) > genem_num:
                genem_list = np.random.choice(genem_list, size=genem_num, replace=False)
            with open(args.path.misc + 'genem_list_' + save_genem_list_name + "_" + str(genem_num) + '.pickle', 'wb') as fp:
                pickle.dump(genem_list, fp)
        print('genem_list_load')
        with open (args.path.misc + 'genem_list_' + save_genem_list_name + "_" + str(genem_num) + '.pickle', 'rb') as fp:
            genem_list = pickle.load(fp)

        genen_set = set(list(range(n_genen)))
        return genem_list, train_record, eval_record, test_record, genen_set, k_list
    else:
        return [None] * 6


def ctr_eval(args, genem_path, sess, model, data, genem_triplet_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    aupr_list = []
    while start + batch_size <= data.shape[0]:
        auc, acc,  f1,aupr = model.eval(sess, get_feed_dict(args, genem_path, model, data, genem_triplet_set, start, start + args.batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        aupr_list.append(aupr)
        start += batch_size

    return auc_list, acc_list, f1_list, aupr_list, float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list)), float(np.mean(aupr_list))


def ctr_eval_case_study(args, genem_path, sess, model, data, genem_triplet_set, genem_history_dict, entity_index_2_name, rela_index_2_name,
    genem_list, genen_set_most_pop, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    aupr_list = []
    nb_size = args.neighbor_sample_size

    if args.SW_stage == 4:
        mixhop_parameter_path = f"{args.path.case_st}{args.log_name}_ep_{str(args.epoch)}_st_{str(args.SW_stage)}.log"
        eval_log_save = open(mixhop_parameter_path, 'w')
        text_space = "*" * 50 + "\n"
        eval_log_save.write(f"{text_space} case_study \n")

        while start + batch_size <= data.shape[0]:
            genem_indices, labels, genen_indices, entities_data, relations_data, importance_list, importance_list_0, importance_list_1 = model.eval_case_study(sess, get_feed_dict(args, genem_path, model, data,
                genem_triplet_set, start, start + args.batch_size))
            for b_i in range(batch_size):

                if genem_indices[b_i] in genem_list and  genen_indices[b_i] in genen_set_most_pop:
                    eval_log_save.write(f"{'*'* 50}\n")

                    eval_log_save.write(f"genem_indices = {genem_indices[b_i]}, genen_indices = {genen_indices[b_i]}, labels = {labels[b_i]}\n")
                    eval_log_save.write(f"{'*'* 20} first_layer  {'*'* 20}\n")
                    eval_log_save.write(f"et_index 0 = {','.join('%s' %dt for dt in entities_data[0][b_i,:].tolist())}\n")
                    eval_log_save.write(f"rela_index 0 = {','.join('%s' %dt for dt in relations_data[0][b_i,:].tolist())}\n")
                    eval_log_save.write(f"et_index 1 = {','.join('%s' %dt for dt in entities_data[1][b_i,:].tolist())}\n")
                    eval_log_save.write(f"{'*'* 20} second_layer  {'*'* 20}\n")

                    for k in range(nb_size):

                        eval_log_save.write(f"entities 0 = {str(k)}\n")
                        eval_log_save.write(f"et_index 0 = {','.join('%s' %dt for dt in [entities_data[1][b_i,k].tolist()])}\n")
                        eval_log_save.write(f"rela_index 0 = {','.join('%s' %dt for dt in relations_data[1][b_i,nb_size*k: nb_size*(k+1)].tolist())}\n")
                        eval_log_save.write(f"et_index 1 = {','.join('%s' %dt for dt in entities_data[2][b_i,nb_size*k: nb_size*(k+1)].tolist())}\n")

                    eval_log_save.write(f"{'*'* 20} entity_relation_name  {'*'* 20}\n")

                    genem_interact_genens = index_2_name_title(genem_history_dict[genem_indices[b_i]], entity_index_2_name)
                    genen_name = entity_index_2_name[str(genen_indices[b_i])] if str(genen_indices[b_i]) in entity_index_2_name else str(genen_indices[b_i])

                    eval_log_save.write(f"genen_name = {genen_name}\n")
                    eval_log_save.write(f"genem_interact_genens = {','.join(genem_interact_genens)}\n")

                    entities_name = [index_2_name_title(et_data[b_i,:], entity_index_2_name) for et_data in entities_data]
                    relation_name = [index_2_name(rela_data[b_i,:], rela_index_2_name) for rela_data in relations_data]
                    eval_log_save.write(f"{'*'* 20} first_layer  {'*'* 20}\n")

                    eval_log_save.write(f"et_index 0 = {','.join('%s' %dt for dt in entities_name[0])}\n")
                    rea_pair = ['rela = %s, enti = %s, att = %s' % (pair[0],pair[1],pair[2]) for pair in zip(relation_name[0], entities_name[1], importance_list_0[b_i,:][0])]
                    rea_pair = '\n'.join(rea_pair)
                    eval_log_save.write(f"er rela pair 0 = {rea_pair}\n")
                    eval_log_save.write(f"{'*'* 20} second_layer  {'*'* 20}\n")

                    for k in range(nb_size):
                        eval_log_save.write(f"entities 0 = {str(k)}\n")
                        eval_log_save.write(f"et_index 0 = {','.join('%s' %dt for dt in [entities_name[1][k]])}\n")
                        rea_pair = ['rela = %s, enti = %s, att = %s' % (pair[0],pair[1],pair[2]) for pair in zip(relation_name[1][nb_size*k: nb_size*(k+1)], entities_name[2][nb_size*k: nb_size*(k+1)], importance_list_1[b_i,k])]
                        rea_pair = '\n'.join(rea_pair)
                        eval_log_save.write(f"er rela pair 1 = {rea_pair}\n")
            start += batch_size

def index_2_name(list_array, dictionary):
    return [dictionary[str(et)] if str(et) in dictionary else str(et) for et in list_array]

def index_2_name_title(list_array, dictionary):
    # print('list_array = ', list_array)
    # print('dictionary = ', dictionary)
    return [dictionary[str(et)] if str(et) in dictionary else str(et) for et in list_array]

def topk_eval(sess, args, genem_triplet_set, model, genem_list, train_record, eval_record, test_record, genen_set, k_list, batch_size, mode = 'test'):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    MAP_list = {k: [] for k in k_list}
    hit_ratio_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    
    for genem in genem_list:
        if mode == 'eval': ref_genem = eval_record
        else: ref_genem = test_record
        if genem in ref_genem:
            test_genen_list = list(genen_set - train_record[genem])
            genen_score_map = dict()
            start = 0
            while start + batch_size <= len(test_genen_list):
                data = []

                genem_list_tmp = [genem] * batch_size
                genen_list = test_genen_list[start:start + batch_size]
                labels_list = [1] * batch_size

                genens, scores = model.get_scores(sess, get_feed_dict_top_k(args, model, genem_list_tmp, genen_list, labels_list, genem_triplet_set))

                for genen, score in zip(genens, scores):
                    genen_score_map[genen] = score
                start += batch_size

            # padding the last incomplete minibatch if exists
            if start < len(test_genen_list):

                genem_list_tmp = [genem] * batch_size
                genen_list = test_genen_list[start:] + [test_genen_list[-1]] * (batch_size - len(test_genen_list) + start)
                labels_list = [1] * batch_size

                genens, scores = model.get_scores(sess, get_feed_dict_top_k(args, model, genem_list_tmp, genen_list, labels_list, genem_triplet_set))


                for genen, score in zip(genens, scores):
                    genen_score_map[genen] = score

            genen_score_pair_sorted = sorted(genen_score_map.genens(), key=lambda x: x[1], reverse=True)
            genen_sorted = [i[0] for i in genen_score_pair_sorted]


            for k in k_list:
                precision_list[k].append(precision_at_k(genen_sorted,ref_genem[genem],k))
                recall_list[k].append(recall_at_k(genen_sorted,ref_genem[genem],k))

            # ndcg
            r_hit = []
            for i in genen_sorted[:k]:
                if i in ref_genem[genem]:
                    r_hit.append(1)
                else:
                    r_hit.append(0)
            for k in k_list:
                ndcg_list[k].append(ndcg_at_k(r_hit,k))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg, None, None


def get_feed_dict(args, genem_path, model, data, genem_triplet_set, start, end):
    feed_dict = {model.genem_indices: data[start:end, 0],
                 model.genen_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}

    for i in range(max(1,args.p_hop)):
        feed_dict[model.memories_h[i]] = [genem_triplet_set[genem][i][0] for genem in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [genem_triplet_set[genem][i][1] for genem in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [genem_triplet_set[genem][i][2] for genem in data[start:end, 0]]

    return feed_dict

def get_feed_dict_top_k(args, model, genem_list, genen, label, head, genem_triplet_set):
    feed_dict = {model.genem_indices: genem_list,
                 model.genen_indices: genen,
                 model.labels: label,model.head_indices: head}

    for i in range(max(1,args.p_hop)):
        feed_dict[model.memories_h[i]] = [genem_triplet_set[genem][i][0] for genem in genem_list]
        feed_dict[model.memories_r[i]] = [genem_triplet_set[genem][i][1] for genem in genem_list]
        feed_dict[model.memories_t[i]] = [genem_triplet_set[genem][i][2] for genem in genem_list]

    return feed_dict

def get_genem_record(data, is_train):
    genem_history_dict = dict()
    for interaction in data:
        genem = interaction[0]
        genen = interaction[1]
        label = interaction[2]
        if label == 1:
            if genem not in genem_history_dict:
                genem_history_dict[genem] = set()
            genem_history_dict[genem].add(genen)
    return genem_history_dict
