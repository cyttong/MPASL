import os
import sys

import tensorflow as tf

import numpy as np
from time import time
from model import MPASL
from train_util import Early_stop_info, Eval_score_info, Train_info_record_sw_emb
from metrics import ndcg_at_k, map_at_k, recall_at_k, hit_ratio_at_k, mrr_at_k, precision_at_k
import pickle
from util import topk_settings, ctr_eval, ctr_eval_case_study, topk_eval

# np.random.seed(1)

def train(args, data, trn_info, show_loss, show_topk):

    n_genem, n_genen, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation, genem_triplet_set = data[7], data[8], data[9]
    genem_path, genem_path_top_k, genen_set_most_pop, genem_history_dict = data[10], data[11], data[12], data[13]
    entity_index_2_name, rela_index_2_name = data[14], data[15]

    
    early_st_info = Early_stop_info(args,show_topk)
    eval_score_info = Eval_score_info()

    interaction_table, offset = get_interaction_table(test_data, n_entity)
    
    # top-K evaluation settings
    genem_list, train_record, eval_record, test_record, genen_set, k_list = topk_settings(args, show_topk, train_data, eval_data, test_data, n_genen, args.save_record_genem_list, args.save_model_name)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        model = MPASL(args, n_genem, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset)
        interaction_table.init.run()

        sess.run(tf.global_variables_initializer())
        trn_info.logger.info('Parameters:' + '\n'.join([v.name for v in tf.global_variables()]))

        stage_wise_var = [v for v in tf.global_variables() if 'STWS' in v.name and 'Adam' not in v.name and 'adam' not in v.name]

        trn_info.logger.info('SW Parameters:' + '\n'.join([v.name for v in stage_wise_var]))
        for SW_varitable in stage_wise_var:
            print("stage_wise variable = ", SW_varitable.name)


        saver = tf.train.Saver(stage_wise_var)

        if args.load_pretrain_emb == True:
            saver.restore(sess, f"{args.path.emb}_sw_para_{args.save_model_name}" + '_parameter')


        for step in range(args.n_epochs):
                # training
            np.random.shuffle(train_data)
            np.random.shuffle(test_data)
            np.random.shuffle(eval_data)
            start = 0
            t_start = time()
            while start +args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess,get_feed_dict(args, genem_path, model, train_data, genem_triplet_set, start, start + args.batch_size))
                start += args.batch_size
            t_flag = time()

        for step in range(args.n_epochs):
                    # training
            np.random.shuffle(train_data)
            np.random.shuffle(test_data)
            np.random.shuffle(eval_data)
            start = 0
            t_start = time()
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess,get_feed_dict(args, genem_path, model, train_data, genem_triplet_set, start,
                                                            start + args.batch_size))
                start += args.batch_size
            t_flag = time()


            # top-K evaluation
            if show_topk:
                precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                    sess, args, genem_triplet_set, model, genem_list, train_record, eval_record, test_record, genen_set_most_pop, k_list, args.batch_size, mode = 'eval')
                n_precision_eval = [round(i, 6) for i in precision]
                n_recall_eval = [round(i, 6) for i in recall]
                n_ndcg_eval = [round(i, 6) for i in ndcg]

                precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                    sess, args, genem_triplet_set, model, genem_list, train_record, eval_record, test_record, genen_set_most_pop, k_list,
                    args.batch_size, mode = 'test')

                n_precision_test = [round(i, 6) for i in precision]
                n_recall_test = [round(i, 6) for i in recall]
                n_ndcg_test = [round(i, 6) for i in ndcg]

                eval_score_info.eval_ndcg_recall_pecision = [n_ndcg_eval, n_recall_eval, n_precision_eval]
                eval_score_info.test_ndcg_recall_pecision = [n_ndcg_test, n_recall_test, n_precision_test]

                trn_info.update_score(step, eval_score_info)
                
                if early_st_info.update_score(step,n_recall_eval[2],sess,model,saver) == True: break
            else:
                # CTR evaluation
                _, _, _, _, auc, acc, f1, aupr= ctr_eval(args, genem_path, sess, model, train_data, genem_triplet_set, args.batch_size)
                eval_score_info.train_auc_acc_f1_aupr = auc, acc, f1, aupr
                _, _, _,_, auc, acc, f1, aupr = ctr_eval(args, genem_path, sess, model, eval_data, genem_triplet_set, args.batch_size)
                eval_score_info.eval_auc_acc_f1_aupr = auc, acc, f1, aupr
                test_auc_list, test_acc_list, test_f1_list, test_aupr_list ,auc, acc, f1, aupr = ctr_eval(args, genem_path, sess, model, test_data, genem_triplet_set, args.batch_size)
                eval_score_info.test_auc_acc_f1_aupr = auc, acc, f1, aupr

                satistic_list = [test_auc_list, test_acc_list, test_f1_list, test_aupr_list]

                trn_info.update_score(step, eval_score_info)

                training_condition =  early_st_info.update_score(step,eval_score_info.eval_st_score(),sess,model,saver, satistic_list= satistic_list)
                if training_condition == "EarlyStopping":
                    if args.attention_cast_st == True: 
                        ctr_eval_case_study(args, genem_path, sess, model, test_data, genem_triplet_set, genem_history_dict, entity_index_2_name,
                        rela_index_2_name, genem_list, genen_set_most_pop, args.batch_size)
                    break

    tf.reset_default_graph()

def get_interaction_table(train_data, n_entity):
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.astype(np.int64)
    values = train_data[:, 2].astype(np.float32)

    interaction_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5)

    return interaction_table, offset



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
                 model.labels: label, model.head_indices: head}

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
