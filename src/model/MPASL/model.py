import tensorflow as tf
from aggregators import SumAggregator_mrh_matrix, ConcatAggregator, NeighborAggregator, PoolAggregator, BiAggregator, TopAggregator, DiAggregator ,Aggregator ,LabelAggregator
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
import numpy as np
import sklearn.metrics as m

class MPASL(object):
    def __init__(self, args, n_genem, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset):
        self._parse_args(args, adj_entity, adj_relation, interaction_table, offset)
        self._build_inputs()
        self._build_model(args,n_genem, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation, interaction_table, offset):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.dataset = args.dataset

        self.load_pretrain_emb = args.load_pretrain_emb
        self.h_hop = args.h_hop
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.p_hop = args.p_hop
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.l2_agg_weight = args.l2_agg_weight
        self.kge_weight = args.kge_weight
        self.lr = args.lr
        self.save_model_name = args.save_model_name
        self.n_mix_hop = args.n_mix_hop
        self.n_memory = args.n_memory
        self.update_genen_emb = args.update_genen_emb
        self.h0_att = args.h0_att
        self.path = args.path
        self.Genem_orient_rela = args.Genem_orient_rela

        self.args = args
        self.act= tf.nn.relu
        #self.emb_update_mode = args.emb_update_mode

        self.aggregator_class = SumAggregator_mrh_matrix
        # self.aggregator_class = ConcatAggregator
        # self.aggregator_class = NeighborAggregator
        # self.aggregator_class = PoolAggregator
        # self.aggregator_class = BiAggregator
        # self.aggregator_class = TopAggregator
        # self.aggregator_class = DiAggregator
        # self.aggregator_class = MLPAggregator
        #self.aggregator_class = GRUAggregator


        if self.args.wide_deep == True: self.agg_fun = self.aggregate_delta_whole
        else: self.agg_fun = self.aggregate

        self.interaction_table = interaction_table
        self.offset = offset
        self.ls_weight = args.ls_weight

        self.W_agg = tf.zeros(shape=[self.dim, self.dim], dtype=tf.float32)
        self.w_agg1 = tf.zeros(shape=[self.dim, self.dim], dtype=tf.float32)
        self.w_agg2 = tf.zeros(shape=[self.dim, 1], dtype=tf.float32)

    def _build_inputs(self):
        self.genem_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='genem_indices')
        self.genen_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='genen_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        self.head_indices = tf.placeholder(tf.int32, shape=[None], name='head_indices')
        self.tail_indices = tf.placeholder(tf.int32, shape=[None], name='tail_indices')
        self.relation_indices = tf.placeholder(tf.int32, shape=[None], name='relation_indices')

        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(max(1,self.p_hop)):
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))


    def save_pretrain_emb_fuc(self,sess, saver):
        saver.save(sess, f"{self.args.path.emb}_sw_para_{self.save_model_name}" + '_parameter')

    def _build_model(self, args, n_genem, n_entity, n_relation):
        self.n_entity = n_entity

        with tf.variable_scope("genem_emb_matrix_STWS"):
            self.genem_emb_matrix = tf.get_variable(
                shape=[n_genem, self.dim], initializer=MPASL.get_initializer(), name='genem_emb_matrix_STWS')

        with tf.variable_scope("entity_emb_matrix_STWS"):
            self.entity_emb_matrix = tf.get_variable(
                shape=[n_entity, self.dim], initializer=MPASL.get_initializer(), name='entity_emb_matrix_STWS')

        with tf.variable_scope("relation_emb_matrix_STWS"):
            self.relation_emb_matrix = tf.get_variable(
                shape=[n_relation,self.dim], initializer=MPASL.get_initializer(), name='relation_emb_matrix_STWS')

        with tf.variable_scope("relation_emb_KGE_matrix_STWS"):
            self.relation_emb_KGE_matrix = tf.get_variable(
                shape=[(n_relation + 1) * 2,self.dim, self.dim], initializer=MPASL.get_initializer(), name='relation_emb_KGE_matrix_STWS')

        self.enti_transfer_matrix_list = []
        self.enti_transfer_bias_list = []

        for n in range(self.n_mix_hop):
            with tf.variable_scope("enti_mlp_matrix"+str(n)):
                self.enti_transfer_matrix = tf.get_variable(
                    shape=[self.dim * (self.h_hop+1), self.dim], initializer=MPASL.get_initializer(), name='transfer_matrix'+str(n))
                self.enti_transfer_bias = tf.get_variable(
                    shape=[self.dim], initializer=MPASL.get_initializer(), name='transfer_bias'+str(n))
                self.enti_transfer_matrix_list.append(self.enti_transfer_matrix)
                self.enti_transfer_bias_list.append(self.enti_transfer_bias)

        with tf.variable_scope("genem_mlp_matrix"):
            if self.args.PS_O_ft == True: genem_mlp_shape = self.p_hop+1
            else: genem_mlp_shape = self.p_hop
            self.genem_mlp_matrix = tf.get_variable(
                shape=[self.dim * (genem_mlp_shape), self.dim], initializer=MPASL.get_initializer(), name='genem_mlp_matrix')
            self.genem_mlp_bias = tf.get_variable(shape=[self.dim], initializer=MPASL.get_initializer()
                                                , name='genem_mlp_bias')
        self.transfer_matrix_list = []
        self.transfer_matrix_bias = []
        for n in range(self.n_mix_hop*self.h_hop+1):
            with tf.variable_scope("transfer_agg_matrix"+str(n)):
                self.transform_matrix = tf.get_variable(name='transfer_agg_matrix'+str(n), shape=[self.dim, self.dim], dtype=tf.float32,
                                                initializer=MPASL.get_initializer())
                self.transform_bias = tf.get_variable(name='transfer_agg_bias'+str(n), shape=[self.dim], dtype=tf.float32,
                                                initializer=MPASL.get_initializer())
                self.transfer_matrix_bias.append(self.transform_bias)
                self.transfer_matrix_list.append(self.transform_matrix)

        with tf.variable_scope("h_emb_genen_mlp_matrix"):
            self.h_emb_genen_mlp_matrix = tf.get_variable(
                shape=[self.dim * 2, 1], initializer=MPASL.get_initializer(), name='h_emb_genen_mlp_matrix')
            self.h_emb_genen_mlp_bias = tf.get_variable(shape=[1], initializer=MPASL.get_initializer()
                                                , name='h_emb_genen_mlp_bias')


        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(max(1,self.p_hop)):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))
            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_KGE_matrix, self.memories_r[i]))
            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))###

        self.head_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.head_indices)
        self.relation_embeddings = tf.nn.embedding_lookup(self.relation_emb_matrix, self.relation_indices)
        self.tail_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.tail_indices)

        # [batch_size, dim]
        entities, relations = self.get_neighbors(self.genen_indices)

        self.entities_data = entities
        self.relations_data = relations

        if self.args.PS_only == True:
            genem_o, transfer_o = self._key_addressing()
            genen_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.genen_indices)

        elif self.args.HO_only == True:
            genem_o = tf.nn.embedding_lookup(self.genem_emb_matrix, self.genem_indices)
            if self.args.Genem_orient_kg_eh == True:  _, transfer_o = self._key_addressing()
            else: transfer_o = [genem_o]
            genen_embeddings, self.aggregators = self.agg_fun(entities, relations, transfer_o)

        else:
            print('MPASL PS and HO')
            genem_o, transfer_o = self._key_addressing()
            if self.args.Genem_orient_kg_eh == False: transfer_o = [tf.nn.embedding_lookup(self.genem_emb_matrix, self.genem_indices)]
            genen_embeddings, self.aggregators = self.agg_fun(args, entities, relations, transfer_o)

        use_inner_product = True
        if use_inner_product:
            # [batch_size]
            self.scores = tf.reduce_sum(genem_o * genen_embeddings, axis=1)

        self.scores_normalized = tf.sigmoid(self.scores)

        self._build_label_smoothness_loss(args, entities, relations, transfer_o)

    def _key_addressing(self):
        def soft_attention_h_set():
            genem_embedding_key = tf.nn.embedding_lookup(self.entity_emb_matrix, self.genem_indices)
            # [batch_size, 1, dim]
            genen = tf.expand_dims(genem_embedding_key, axis=1)
            # [batch_size, n_memory, dim]
            genen = tf.tile(genen, [1, self.h_emb_list[0].shape[1], 1])
            h_emb_genen = [self.h_emb_list[0],genen]
            # [batch_size, n_memory, 2 * dim]
            h_emb_genen = tf.concat(h_emb_genen, 2)

            # [-1 , dim * 2]
            h_emb_genen = tf.reshape(h_emb_genen,[-1,self.dim * 2])
            probs = tf.squeeze(tf.matmul(h_emb_genen, self.h_emb_genen_mlp_matrix), axis=-1) + self.h_emb_genen_mlp_bias

            # [batch_size, n_memory]
            probs = tf.reshape(probs,[-1,self.h_emb_list[0].shape[1]])
            probs_normalized = tf.nn.softmax(probs)
            # [batch_size, n_memory,1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, 1, dim]
            genem_h_set = tf.reduce_sum(self.h_emb_list[0] * probs_expanded, axis=1)

            return genem_h_set

        genen_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.genen_indices)
        o_list = []

        if self.args.PS_O_ft == True:
            genem_h_set = soft_attention_h_set()
            o_list.append(genem_h_set)

        transfer_o = []

        for hop in range(self.p_hop): # v Ri hi
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)
            # [batch_size, n_memory, dim]
            v = tf.expand_dims(genen_embeddings, axis=2)

            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)
            o_list.append(o)

        o_list = tf.concat(o_list, -1)
        if self.args.PS_O_ft == True:
            genem_o = tf.matmul(tf.reshape(o_list,[-1,self.dim * (self.p_hop+1)]), self.genem_mlp_matrix) + self.genem_mlp_bias
        else:
            genem_o = tf.matmul(tf.reshape(o_list,[-1,self.dim * (self.p_hop)]), self.genem_mlp_matrix) + self.genem_mlp_bias

        transfer_o.append(genem_o)

        return genem_o, transfer_o

    def update_item_embedding(self, genen_embeddings, o):
        if self.genen_update_mode == "replace":
            genen_embeddings = o
        elif self.genen_update_mode == "plus":
            genen_embeddings = genen_embeddings + o
        elif self.genen_update_mode == "replace_transform":
            genen_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.genen_update_mode == "plus_transform":
            genen_embeddings = tf.matmul(genen_embeddings + o, self.transform_matrix)
        elif self.genen_update_mode == "agg_attention":
            genen_embeddings = self.agg_attention(genen_embeddings)
        else:
            raise Exception("Unknown nodeb updating mode: " + self.genen_update_mode)
        return genen_embeddings


    def get_neighbors(self, seeds):

        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]

        relations = []
        n = self.n_neighbor
        for i in range(self.n_mix_hop*self.h_hop):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, n])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, n])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            n *= self.n_neighbor
        return entities, relations


    def aggregate_delta_whole(self, args, entities, relations, transfer_o):
        # print('aggregate_delta_whole ===')
        genem_query = transfer_o[0]
        print('MPASL aggregate_delta_whole')
        aggregators = []  # store all aggregators
        mix_hop_res = []

        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        if self.args.Genem_orient == True:
            print('Genem_orient')
            for index in range(len(transfer_o)):
                transfer_o[index] = tf.expand_dims(transfer_o[index], axis=1)
            for index in range(len(transfer_o)):
                for e_i in range(len(entity_vectors)):
                    # [b,1,dim]
                    n_entities = entity_vectors[e_i] + transfer_o[index]
                    # [-1,dim]
                    n_entities = tf.matmul(tf.reshape(n_entities, [-1,self.dim]), self.transfer_matrix_list[e_i]) + self.transfer_matrix_bias[e_i]
                    # [b,n,dim]
                    entity_vectors[e_i] = tf.reshape(n_entities, [self.batch_size, entity_vectors[e_i].shape[1],self.dim])
                    # [b,?*n,dim]
                    transfer_o[index] = tf.tile(transfer_o[index],[1,self.n_neighbor,1])

        for n in range(self.n_mix_hop):
            mix_hop_tmp = []
            mix_hop_tmp.append(entity_vectors)
            for i in range(self.h_hop):
                aggregator = self.aggregator_class(args,self.save_model_name,self.batch_size, self.dim, name = str(i)+'_'+str(n), Genem_orient_rela = self.Genem_orient_rela)
                aggregators.append(aggregator)
                entity_vectors_next_iter = []

                if i == 0: self.importance_list = []
                for hop in range(self.h_hop*self.n_mix_hop-(self.h_hop*n+i)):
                    shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                    shape_r = [self.batch_size, -1, self.n_neighbor, self.dim]
                    vector, probs_normalized = aggregator(self_vectors=entity_vectors[hop],
                                        neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                        neighbor_relations=tf.reshape(relation_vectors[hop], shape_r),
                                        genem_embeddings=genem_query,
                                        masks=None)

                    #if i == 0: self.importance_list.append(probs_normalized)
                    entity_vectors_next_iter.append(vector)
                entity_vectors = entity_vectors_next_iter
                mix_hop_tmp.append(entity_vectors)

            entity_vectors = []
            for mip_hop in zip(*mix_hop_tmp):
                mip_hop = tf.concat(mip_hop, -1)
                mip_hop = tf.matmul(tf.reshape(mip_hop,[-1,self.dim * (self.h_hop+1)]), self.enti_transfer_matrix_list[n]) + self.enti_transfer_bias_list[n]
                mip_hop = tf.reshape(mip_hop,[self.batch_size,-1,self.dim])
                entity_vectors.append(mip_hop)
                if len(entity_vectors) == (self.n_mix_hop-(n+1))*self.h_hop+1:  break

        mix_hop_res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])


        return mix_hop_res, aggregators


    def _build_label_smoothness_loss(self, args, entities, relations, transfer_o):

        genem_query = tf.nn.embedding_lookup(self.entity_emb_matrix, self.genem_indices)
        entity_labels = []

        reset_masks = []
        holdout_item_for_genem = None

        for entities_per_iter in entities:
            # [batch_size, 1]
            genems = tf.expand_dims(self.genem_indices, 1)

            genem_entity_concat = tf.maximum(genems * self.offset, entities_per_iter)

            if holdout_item_for_genem is None:
                holdout_item_for_genem = genem_entity_concat

            # [batch_size, n_neighbor^i]
            initial_label = self.interaction_table.lookup(genem_entity_concat)
            holdout_mask = tf.cast(holdout_item_for_genem - genem_entity_concat,
                                       tf.bool)
            reset_mask = tf.cast(initial_label - tf.constant(0.5), tf.bool)
            reset_mask = tf.logical_and(reset_mask, holdout_mask)
            initial_label = tf.cast(holdout_mask, tf.float32) * initial_label + tf.cast(
                    tf.logical_not(holdout_mask), tf.float32) * tf.constant(0.5)
            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        reset_masks = reset_masks[:-1]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        aggregator = LabelAggregator(args, self.save_model_name, self.batch_size, self.dim)

        for n in range(self.n_mix_hop):
            mix_hop_tmp = []
            mix_hop_tmp.append(entity_labels)
            for i in range(self.h_hop):
                entity_labels_next_iter = []
                for hop in range(self.h_hop * self.n_mix_hop - (self.h_hop * n + i)):
                    shape = [self.batch_size, -1, self.n_neighbor]  #
                    shape_r = [self.batch_size, -1, self.n_neighbor, self.dim]
                    vector = aggregator(self_vectors=entity_labels[hop],
                                            neighbor_vectors=tf.reshape(entity_labels[hop + 1], shape),
                                            neighbor_relations=tf.reshape(relation_vectors[hop], shape_r),
                                            genem_embeddings=genem_query,
                                            masks=reset_masks[hop])
                    entity_labels_next_iter.append(vector)
                entity_labels = entity_labels_next_iter

        self.predicted_labels = tf.squeeze(entity_labels[0], axis=-1)



    def aggregate(self,args, entities, relations, transfer_o):

        genem_query = transfer_o[0]

        print('aggregate agg method')
        aggregators = []
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        if self.args.Genem_orient == True:
            for index in range(len(transfer_o)):
                # [b,1,dim]
                transfer_o[index] = tf.expand_dims(transfer_o[index], axis=1)
            for index in range(len(transfer_o)):
                for e_i in range(len(entity_vectors)):
                    n_entities = entity_vectors[e_i] + transfer_o[index]
                    # [-1,dim]
                    n_entities = tf.matmul(tf.reshape(n_entities, [-1,self.dim]), self.transfer_matrix_list[e_i]) + self.transfer_matrix_bias[e_i]
                    entity_vectors[e_i] = tf.reshape(n_entities, [self.batch_size, entity_vectors[e_i].shape[1],self.dim])
                    # [b,?*n,dim]
                    transfer_o[index] = tf.tile(transfer_o[index],[1,self.n_neighbor,1])

        tf.compat.v1.disable_eager_execution()
        tf.reset_default_graph()
        for i in range(self.h_hop):

            aggregator = self.aggregator_class(self.save_model_name,self.batch_size, self.dim, name = i)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.h_hop - i):
                shape = [self.batch_size, entity_vectors[hop].shape[1], self.n_neighbor, self.dim]
                shape_r = [self.batch_size, entity_vectors[hop].shape[1], self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape_r),
                                    genem_embeddings=genem_query,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    #loss
    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = 0
        for hop in range(self.p_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))

        self.l2_loss += tf.nn.l2_loss(self.relation_emb_matrix) 

        self.l2_agg_loss = 0
        
        self.l2_agg_loss += tf.nn.l2_loss(self.genem_emb_matrix)
        if self.args.PS_only != True:
            for aggregator in self.aggregators:
                self.l2_agg_loss += tf.nn.l2_loss(aggregator.weights)
                self.l2_agg_loss += tf.nn.l2_loss(aggregator.mrh_weights)

                # self.l2_agg_loss += tf.nn.l2_loss(aggregator.weights_1)
                # self.l2_agg_loss += tf.nn.l2_loss(aggregator.mrh_weights_1)
                # self.l2_agg_loss += tf.nn.l2_loss(aggregator.weights_2)
                # self.l2_agg_loss += tf.nn.l2_loss(aggregator.mrh_weights_2)

        for n in range(self.n_mix_hop):
            self.l2_agg_loss += tf.nn.l2_loss(self.enti_transfer_matrix_list[n]) + tf.nn.l2_loss(self.enti_transfer_bias_list[n])
        
        if self.p_hop > 0:
            self.l2_loss += tf.nn.l2_loss(self.genem_mlp_matrix) + tf.nn.l2_loss(self.genem_mlp_bias)
            self.l2_loss += tf.nn.l2_loss(self.transform_matrix) + tf.nn.l2_loss(self.transform_bias)

            for n in range(self.h_hop+1):
                self.l2_loss += tf.nn.l2_loss(self.transfer_matrix_list[n]) + tf.nn.l2_loss(self.transfer_matrix_bias[n])

        self.l2_loss += tf.nn.l2_loss(self.h_emb_genen_mlp_matrix) +  tf.nn.l2_loss(self.h_emb_genen_mlp_bias)

        self.ls_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.predicted_labels))

        self.loss = self.base_loss + self.l2_weight * self.l2_loss + self.l2_agg_weight * self.l2_agg_loss + self.ls_weight * self.ls_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        p, r, t = precision_recall_curve(y_true=labels, probas_pred=scores)
        aupr = m.auc(r, p)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        acc = np.mean(np.equal(scores, labels))
        return auc, acc, f1, aupr

    def eval_case_study(self, sess, feed_dict):
        genem_indices, labels, genen_indices, head_indices, entities_data, relations_data, importance_list, importance_list_0, importance_list_1 = sess.run([self.genem_indices, self.labels, self.genen_indices,self.head_indices,
            self.entities_data, self.relations_data, self.importance_list_0, self.importance_list_1], feed_dict)

        return genem_indices, labels, genen_indices, head_indices, entities_data, relations_data, importance_list_0, importance_list_1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.genen_indices, self.scores_normalized], feed_dict)


    def agg_attention(self, genen_embeddings):
        genen_embeddings = tf.expand_dims(genen_embeddings, axis=1)

        Wemb1 = tf.matmul(genen_embeddings, self.W_agg)
        genen_scores = tf.matmul(tf.nn.tanh(Wemb1), self.w_agg1)

        alpha1 = tf.nn.softmax(genen_scores)
        alpha1 = tf.transpose(alpha1, [0, 2, 1])

        genen_embeddings = tf.matmul(alpha1, genen_embeddings)

        genen_embeddings = tf.matmul(genen_embeddings, self.w_agg2)
        genen_embeddings = tf.squeeze(genen_embeddings)

        return genen_embeddings
