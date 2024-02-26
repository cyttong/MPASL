import tensorflow as tf
from abc import abstractmethod
#import torch.nn as nn
#import torch

LAYER_IDS = {}

# tf.set_random_seed(1)

def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, args, save_model_name, batch_size, dim, dropout, act, name):

        layer = self.__class__.__name__.lower()
        name = layer + '_'+save_model_name+'_' + str(name)
        # print('name = ',name)
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

        self.n_neighbor = args.neighbor_sample_size
        self.top = self.n_neighbor //2
        self.att_type = args.att_type
        self.agg = args.agg

        self.attention = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(self.dim, activation='relu'),
             tf.keras.layers.Dense(self.dim, activation='relu'),
             tf.keras.layers.Dense(1)]
        )
    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings, masks):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings,masks)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings, masks):
        pass

    def _mix_neighbor_vectors(self, self_vectors,neighbor_vectors, neighbor_relations, genem_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            genem_embeddings = tf.reshape(genem_embeddings, [neighbor_vectors.shape[0], 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            genem_relation_scores = tf.reduce_mean(genem_embeddings * neighbor_relations, axis=-1)
            genem_relation_scores = self.attention(genem_relation_scores)
            genem_relation_scores_normalized = tf.nn.softmax(genem_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1] shape_n = [self.batch_size, -1, self.n_neighbor, self.dim]
            genem_relation_scores_normalized = tf.expand_dims(genem_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(genem_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated, genem_relation_scores


    def _mix_neighbor_vectors_mrv(self, self_vectors,neighbor_vectors, neighbor_relations, genem_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            genem_embeddings = tf.reshape(genem_embeddings, [neighbor_vectors.shape[0], 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            genem_relation_scores = tf.reduce_mean(genem_embeddings * neighbor_relations, axis=-1)
            genem_relation_scores_normalized = tf.nn.softmax(genem_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1] shape_n = [self.batch_size, -1, self.n_neighbor, self.dim]
            genem_relation_scores_normalized = tf.expand_dims(genem_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(genem_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated        

class SumAggregator_mrh_matrix(Aggregator):
    def __init__(self, args,save_model_name, batch_size, dim, dropout=0., act=tf.nn.relu, name=None, Genem_orient_rela = True):
        super(SumAggregator_mrh_matrix, self).__init__(args,save_model_name,batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name+'_wights'):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(seed = 500), name='weights')
        with tf.variable_scope(self.name+'_bias'):
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name+'_mrh_wights'):
            self.mrh_weights = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed = 500), name='weights')
        with tf.variable_scope(self.name+'_mrh_bias'):
            self.mrh_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

        self.Genem_orient_rela = Genem_orient_rela
        self.act = tf.nn.relu

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings, masks):
        # [batch_size, -1, dim]
        if self.Genem_orient_rela == True:
            print('self.Genem_orient_rela  = ', self.Genem_orient_rela, '_mix_neighbor_vectors_mrh')
            neighbors_agg, probs_normalized = self._mix_neighbor_vectors(self_vectors,neighbor_vectors, neighbor_relations,
                                                       genem_embeddings)
        else:
            print('self.Genem_orient_rela  = ', self.Genem_orient_rela, '_mix_neighbor_vectors_no_mr')
            neighbors_agg = self._mix_neighbor_vectors_no_mr(self_vectors, genem_embeddings, neighbor_vectors, neighbor_relations)
            probs_normalized = None
        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        # return output
        return self.act(output), probs_normalized


    def _mix_neighbor_vectors_mrh(self, self_vectors, genem_embeddings, neighbor_vectors, neighbor_relations):

        # [batch_size, 1, 1, dim]
        genem_embeddings = tf.reshape(genem_embeddings, [self.batch_size, 1, 1, self.dim])
        genem_embeddings = tf.tile(genem_embeddings, multiples=[1, neighbor_relations.get_shape()[1], neighbor_relations.get_shape()[2], 1])


        # [batch_size, -1, 1, dim]
        self_vectors = tf.expand_dims(self_vectors, axis=2)
        self_vectors = tf.tile(self_vectors, multiples=[1, 1, neighbor_relations.get_shape()[2], 1])

        # [batch_size, -1, -1, dim * 4]
        mrh_matrix = [genem_embeddings, neighbor_relations, self_vectors]
        mrh_matrix = tf.concat(mrh_matrix, -1)


        # [-1, 1]
        mrh_matrix = tf.matmul(tf.reshape(mrh_matrix,[-1, 3 * self.dim]), self.mrh_weights)

        probs = tf.reshape(mrh_matrix,[neighbor_vectors.get_shape()[0],neighbor_vectors.get_shape()[1],neighbor_vectors.get_shape()[2]])

        #probs = self.attention(probs)
        # [batch_size, -1, n_memory]
        probs_normalized = tf.nn.softmax(probs)
        # [batch_size,-1, n_memory, 1]
        probs_expanded = tf.expand_dims(probs_normalized, axis= -1)

        # [batch_size, -1, n_memory]
        neighbors_aggregated = tf.reduce_mean(probs_expanded * neighbor_vectors, axis=2)

        return neighbors_aggregated, probs_normalized

    def _mix_neighbor_vectors_no_ur(self, self_vectors, genem_embeddings, neighbor_vectors, neighbor_relations):

        neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)
        return neighbors_aggregated

class ConcatAggregator(Aggregator):
    def __init__(self, args, save_model_name, batch_size, dim, dropout=0., act=tf.nn.relu,  name=None,Genem_orient_rela=True):
        super(ConcatAggregator, self).__init__(args,save_model_name, batch_size, dim, dropout, act, name)

        #with tf.variable_scope(self.name):
        with tf.variable_scope(self.name + '_weights', reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        with tf.variable_scope(self.name + '_bias', reuse=tf.AUTO_REUSE):
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name+'_mrh_wights',reuse=tf.AUTO_REUSE):
            self.mrh_weights = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed=500), name='weights')
        with tf.variable_scope(self.name+'_mrh_bias',reuse=tf.AUTO_REUSE):
            self.mrh_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

        self.Genem_orient_rela = Genem_orient_rela_orient_rela
        self.act = tf.nn.relu
        #self.vars = [self.weights]

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings, masks):
        # [batch_size, -1, dim]
        if self.Genem_orient_rela == True:
            print('self.Genem_orient_rela  = ', self.Genem_orient_rela, '_mix_neighbor_vectors_mrh')
            neighbors_agg = self._mix_neighbor_vectors(self_vectors,neighbor_vectors, neighbor_relations,
                                                       genem_embeddings)
        else:
            print('self.Genem_orient_rela  = ', self.Genem_orient_rela, '_mix_neighbor_vectors_no_mr')
            neighbors_agg = self._mix_neighbor_vectors_no_mr(self_vectors, genem_embeddings, neighbor_vectors, neighbor_relations)
            probs_normalized = None

        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)



class NeighborAggregator(Aggregator):
    def __init__(self, args,save_model_name, batch_size, dim, dropout=0., act=tf.nn.relu,  name=None,  Genem_orient_rela=True):
        super(NeighborAggregator, self).__init__(args,save_model_name, batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name + '_weights', reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        with tf.variable_scope(self.name + '_bias', reuse=tf.AUTO_REUSE):
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name + '_mrh_wights', reuse=tf.AUTO_REUSE):
            self.mrh_weights = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed=500),
                name='weights')  # seed=1
        with tf.variable_scope(self.name + '_mrh_bias', reuse=tf.AUTO_REUSE):
            self.mrh_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

        self.Genem_orient_rela = Genem_orient_rela
        self.act = tf.nn.relu

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors,neighbor_vectors, neighbor_relations, genem_embeddings)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class TopAggregator(Aggregator):
    def __init__(self, args,save_model_name, batch_size, dim, dropout=0., act=tf.nn.relu,name=None, Genem_orient_rela=True):
        super(TopAggregator, self).__init__(args,save_model_name, batch_size, dim, dropout, act,  name)

        with tf.variable_scope(self.name + '_weights', reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        with tf.variable_scope(self.name + '_bias', reuse=tf.AUTO_REUSE):
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name + '_mrh_wights', reuse=tf.AUTO_REUSE):
            self.mrh_weights = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed=500),
                name='weights')  # seed=1
        with tf.variable_scope(self.name + '_mrh_bias', reuse=tf.AUTO_REUSE):
            self.mrh_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

        self.Genem_orient_rela = Genem_orient_rela
        self.act = tf.nn.relu


    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


    def _mix_neighbor_vectors(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            genem_embeddings = tf.reshape(genem_embeddings, [self.batch_size, 1, 1, self.dim])
            self_vectors = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])


            # [batch_size, -1, n_neighbor]
            genem_relation_scores = tf.reduce_mean(genem_embeddings * neighbor_relations, axis=-1)

            genem_relation_scores = self.attention(genem_relation_scores)

            self_relation_scores=tf.reduce_sum(self_vectors*neighbor_relations,axis=-1)
            self_neighbor_scores=tf.reduce_sum(self_vectors*neighbor_relations,axis=-1)


            genem_relation_scores_normalized = tf.nn.softmax(genem_relation_scores, dim=-1)

            genem_relation_scores_normalized = tf.expand_dims(genem_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = genem_relation_scores_normalized * neighbor_vectors
            neighbors_aggregated=tf.transpose(neighbors_aggregated,[0,1,3,2])
            neighbors_aggregated = tf.nn.top_k(neighbors_aggregated,k=self.top)[0]
            neighbors_aggregated=tf.reduce_mean(neighbors_aggregated,axis=-1)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class DiAggregator(Aggregator):
    def __init__(self, args,save_model_name, batch_size, dim, dropout=0., act=tf.nn.relu, name=None, Genem_orient_rela=True):
        super(DiAggregator, self).__init__(args,save_model_name, batch_size, dim, dropout, act,name)

        #with tf.variable_scope(self.name):
        with tf.variable_scope(self.name + '_weights', reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        with tf.variable_scope(self.name + '_bias', reuse=tf.AUTO_REUSE):
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name + '_mrh_wights', reuse=tf.AUTO_REUSE):
            self.mrh_weights = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed=500),
                name='weights')
        with tf.variable_scope(self.name + '_mrh_bias', reuse=tf.AUTO_REUSE):
            self.mrh_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

        self.Genem_orient_rela = Genem_orient_rela
        self.act = tf.nn.relu

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = tf.reduce_mean(neighbor_vectors,axis=2)

        # [-1, dim]
        output = tf.reshape(self_vectors+neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class PoolAggregator(Aggregator):
    def __init__(self, args,save_model_name, batch_size, dim, dropout=0., act=tf.nn.relu, name=None, Genem_orient_rela=True):
        super(PoolAggregator, self).__init__(args,save_model_name, batch_size, dim, dropout, act, name)

        #with tf.variable_scope(self.name):
        with tf.variable_scope(self.name + '_weights', reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable(
                shape=[self.dim , self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        with tf.variable_scope(self.name + '_bias', reuse=tf.AUTO_REUSE):
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name+'_mrh_wights',reuse=tf.AUTO_REUSE):
            self.mrh_weights = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed=500), name='weights')
        with tf.variable_scope(self.name+'_mrh_bias',reuse=tf.AUTO_REUSE):
            self.mrh_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

        self.Genem_orient_rela = Genem_orient_rela
        self.act = tf.nn.relu

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings)

        # [batch_size, -1, dim * 2]
        output = tf.maximum(self_vectors , neighbors_agg)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim ])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class BiAggregator(Aggregator):
    def __init__(self, args,save_model_name, batch_size, dim, dropout=0., act=tf.nn.relu, name=None,Genem_orient_rela=True):
        super(BiAggregator, self).__init__(args,save_model_name, batch_size, dim, dropout, act, name)

        # with tf.variable_scope(self.name):
        with tf.variable_scope(self.name + '_weights_1', reuse=tf.AUTO_REUSE):
            self.weights_1 = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        with tf.variable_scope(self.name + '_bias_1', reuse=tf.AUTO_REUSE):
            self.bias_1 = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name + '_mrh_wights_1', reuse=tf.AUTO_REUSE):
            self.mrh_weights_1 = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed=500),
                name='weights')  # seed=1
        with tf.variable_scope(self.name + '_mrh_bias_1', reuse=tf.AUTO_REUSE):
            self.mrh_bias_1 = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name + '_weights_2', reuse=tf.AUTO_REUSE):
            self.weights_2 = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        with tf.variable_scope(self.name + '_bias_2', reuse=tf.AUTO_REUSE):
            self.bias_2 = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name + '_mrh_wights_2', reuse=tf.AUTO_REUSE):
            self.mrh_weights_2 = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed=500),
                name='weights')
        with tf.variable_scope(self.name + '_mrh_bias_2', reuse=tf.AUTO_REUSE):
            self.mrh_bias_2 = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

        self.Genem_orient_rela = Genem_orient_rela
        self.act = tf.nn.relu


    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, genem_embeddings)

        # [-1, dim]
        '''
        output_1 = tf.reshape(neighbors_agg, [-1, self.dim])
        output_1 = tf.nn.dropout(output_1, keep_prob=1 - self.dropout)
        output_1 = tf.matmul(output_1, self.weights_1) + self.bias_1

        # [batch_size, -1, dim]
        output_1 = tf.reshape(output_1, [self.batch_size, -1, self.dim])
'''
        output_1 = tf.reshape(tf.multiply(self_vectors, neighbors_agg), [-1, self.dim])
        output_1 = tf.nn.dropout(output_1, keep_prob=1 - self.dropout)
        output_1 = tf.matmul(output_1, self.weights_1) + self.bias_1

        # [batch_size, -1, dim]
        output_1 = tf.reshape(output_1, [self.batch_size, -1, self.dim])

        # [-1, dim]
        output_2 = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output_2 = tf.nn.dropout(output_2, keep_prob=1 - self.dropout)
        output_2 = tf.matmul(output_2, self.weights_2) + self.bias_2

        # [batch_size, -1, dim]
        output_2 = tf.reshape(output_2, [self.batch_size, -1, self.dim])

        return self.act(output_1) + self.act(output_2)


class LabelAggregator(Aggregator):
    def __init__(self,args,save_model_name, batch_size, dim,  name=None):
        super(LabelAggregator, self).__init__(args,save_model_name, batch_size, dim, 0., None, name)

    def _call(self, self_labels, neighbor_labels, neighbor_relations, genem_embeddings, masks):
        # [batch_size, 1, 1, dim]
        genem_embeddings = tf.reshape(genem_embeddings, [self.batch_size, 1, 1, self.dim])

        # [batch_size, -1, n_neighbor]
        genem_relation_scores = tf.reduce_mean(genem_embeddings * neighbor_relations, axis=-1)
        genem_relation_scores_normalized = tf.nn.softmax(genem_relation_scores, dim=-1)

        # [batch_size, -1]
        neighbors_aggregated = tf.reduce_mean(genem_relation_scores_normalized * neighbor_labels, axis=-1)

        output = tf.cast(masks, tf.float32) * self_labels + tf.cast(
            tf.logical_not(masks), tf.float32) * neighbors_aggregated

        return output
