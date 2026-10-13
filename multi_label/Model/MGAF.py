import tensorflow as tf
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MGAF(object):
    def __init__(self, data_config, args):
        self._parse_args(data_config, args)
        self._build_inputs()
        self._init_mgaf_module()
        self.weights = self._build_weights()
        self._build_model_phase_I()
        self._create_DNN_prediction()
        self._build_model_phase_II()
        self._build_loss_phase_II()
        self._build_total_loss()
        self._statistics_params()
        self._setup_validation()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        if len(coo.data) == 0:
            indices = np.array([[0, 0]], dtype=np.int64)
            values = np.array([0.], dtype=np.float32)
            sp = tf.SparseTensor(indices, values, coo.shape)
            sp = tf.sparse.reorder(sp)
            return sp
        indices = np.mat([coo.row, coo.col]).transpose().astype(np.int64)
        values = coo.data.astype(np.float32)
        sp = tf.SparseTensor(indices, values, coo.shape)
        try:
            sp = tf.sparse.reorder(sp)
        except Exception:
            pass
        return sp

    def _sparse_adj_list_process(self):
        """处理稀疏邻接矩阵列表，将其转换为稀疏张量"""
        sparse_adj_list = []
        for adj in self.all_sparse_adj_list:
            convert_adj = self._convert_sp_mat_to_sp_tensor(adj)  # convert to tensor
            sparse_adj_list.append(convert_adj)
        return sparse_adj_list

    def _parse_args(self, data_config, args):
        # argument settings
        self.model_type = 'mgaf'

        self.n_drug1 = data_config['n_drugs']
        self.n_drug2 = self.n_drug1
        self.n_relations = int(data_config['n_relations'])
        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']
        self.n_labels = data_config['n_labels']  # 新增：标签数量

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.dim = args.embed_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.alg_type = args.alg_type
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.verbose = args.verbose
        self.margin = args.margin
        self.B = args.B

        self.dnn_size = eval(args.dnn_size)

        self.all_sparse_adj_list = data_config['sparse_adj_list']  # list of scipy.sparse adjacency matrices

        # 注意：这个方法现在已经被提前定义了
        self.sparse_adj_list = self._sparse_adj_list_process()

        # new options
        # dropedge rate for regularization
        self.dropedge_rate = getattr(args, 'dropedge_rate', 0.1)
        # focal loss gamma and alpha
        self.focal_gamma = getattr(args, 'focal_gamma', 2.0)
        self.focal_alpha = getattr(args, 'focal_alpha', 0.25)

        # 固定阈值0.5
        self.fixed_threshold = 0.5

    def _build_inputs(self):
        # placeholder definition
        self.drugs = tf.placeholder(tf.int32, shape=(None,))
        self.pos_drugs = tf.placeholder(tf.int32, shape=(None,))
        self.neg_drugs = tf.placeholder(tf.int32, shape=(None,))

        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

        # message dropout (adopted on the convolution operations).
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        # 修改：从one-hot标签改为多标签向量
        self.multi_label = tf.placeholder(tf.float32, shape=(None, self.n_labels))  # 多标签向量

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # global step / epoch for adaptive momentum schedule
        self.global_step = tf.placeholder(tf.int32, shape=(), name='global_step')

    def _init_mgaf_module(self):
        self.mgaf_in_dim = self.weight_size[-1]
        self.mgaf_hidden = 128
        self.mgaf_kge_dim = 64
        self.mgaf_rel_total = self.n_relations
        self.mgaf_heads_out_feat_params = [64, 32]
        self.mgaf_blocks_params = [4, 2]

    def _build_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['re_drug1_embed'] = tf.Variable(initializer([self.n_drug1, self.emb_dim]), name='re_drug1_embed')
        all_weights['re_drug2_embed'] = tf.Variable(initializer([self.n_drug2, self.emb_dim]), name='re_drug2_embed')
        all_weights['im_drug1_embed'] = tf.Variable(initializer([self.n_drug1, self.emb_dim]), name='im_drug1_embed')
        all_weights['im_drug2_embed'] = tf.Variable(initializer([self.n_drug2, self.emb_dim]), name='im_drug2_embed')
        print('using xavier initialization')

        all_weights['re_relation_embed'] = tf.Variable(initializer([self.n_relations, self.dim]),
                                                       name='re_relation_embed')
        all_weights['im_relation_embed'] = tf.Variable(initializer([self.n_relations, self.dim]),
                                                       name='im_relation_embed')
        all_weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.dim]),
                                                    name='relation_embed')
        all_weights['relation_mapping'] = tf.Variable(initializer([self.n_relations, 2 * self.dim]),
                                                      name='relation_mapping')
        all_weights['relation_matrix'] = tf.Variable(initializer([self.B, self.dim, self.dim]),
                                                     name='relation_matrix')
        relation_initial = np.random.randn(self.n_relations, self.B).astype(np.float32)

        all_weights['alpha'] = tf.Variable(relation_initial, name='arb')
        relation_1_att = np.random.randn(self.n_relations, self.n_drug1, 1).astype(np.float32)
        relation_2_att = np.random.randn(self.n_relations, self.n_drug2, 1).astype(np.float32)
        all_weights['relation_1_att'] = tf.Variable(relation_1_att, name='relation_1_att')
        all_weights['relation_2_att'] = tf.Variable(relation_2_att, name='relation_2_att')

        self.weight_size_list = [self.emb_dim * 2] + self.weight_size

        all_weights['W_mlp_0'] = tf.Variable(
            initializer([384, self.weight_size_list[1]]), name='W_mlp_0')
        all_weights['b_mlp_0'] = tf.Variable(
            initializer([1, self.weight_size_list[1]]), name='b_mlp_0')

        # 修复：计算正确的输入维度
        # final_feature的维度是 fused_feat的维度 + 1 (bilinear_interaction)
        # fused_feat的维度是 self.weight_size_list[-1] (即最后一个隐藏层的维度)
        dnn_input_dim = self.weight_size_list[-1] + 1  # 应该是200 + 1 = 201
        print(f"DNN input dimension: {dnn_input_dim}")

        # 验证dnn_size配置
        print(f"DNN size configuration: {self.dnn_size}")
        print(f"Number of labels: {self.n_labels}")

        # 修复：重新构建DNN权重，确保维度匹配
        # 假设dnn_size = [201, 128, 64, 24]
        # 这意味着：
        # - 输入层: [201, 128]
        # - 隐藏层1: [128, 64]
        # - 输出层: [64, n_labels]

        # 但为了确保正确，我们重新构建权重列表
        # 创建维度列表：从输入到输出
        layer_dims = [dnn_input_dim] + list(self.dnn_size[1:])  # [201, 128, 64, 24]

        print(f"Building DNN with layer dimensions: {layer_dims}")

        # 构建权重
        for i in range(len(layer_dims) - 1):
            input_dim = layer_dims[i]
            output_dim = layer_dims[i + 1] if i < len(layer_dims) - 2 else self.n_labels

            weight_name = f'nn_w{i + 1}'
            bias_name = f'nn_b{i + 1}'

            print(f"Layer {i + 1}: {input_dim} -> {output_dim}")

            all_weights[weight_name] = tf.Variable(
                initializer([input_dim, output_dim]),
                name=weight_name
            )
            all_weights[bias_name] = tf.Variable(
                initializer([1, output_dim]),
                name=bias_name
            )

        # mgaf attention weights
        all_weights['mgaf_q_w'] = tf.Variable(initializer([self.mgaf_in_dim, self.mgaf_hidden]), name='mgaf_q_w')
        all_weights['mgaf_k_w'] = tf.Variable(initializer([self.mgaf_in_dim, self.mgaf_hidden]), name='mgaf_k_w')
        all_weights['mgaf_v_w'] = tf.Variable(initializer([self.mgaf_in_dim, self.mgaf_hidden]), name='mgaf_v_w')

        # inter attention
        all_weights['mgaf_inter_q_w'] = tf.Variable(initializer([self.mgaf_in_dim, self.mgaf_hidden]),
                                                    name='mgaf_inter_q_w')
        all_weights['mgaf_inter_k_w'] = tf.Variable(initializer([self.mgaf_in_dim, self.mgaf_hidden]),
                                                    name='mgaf_inter_k_w')
        all_weights['mgaf_inter_v_w'] = tf.Variable(initializer([self.mgaf_in_dim, self.mgaf_hidden]),
                                                    name='mgaf_inter_v_w')
        pool_dim = int(self.mgaf_hidden * 2)
        all_weights['mgaf_pool_score'] = tf.Variable(initializer([pool_dim, 1]), name='mgaf_pool_score')
        all_weights['mgaf_proj_w'] = tf.Variable(initializer([pool_dim, self.mgaf_hidden]), name='mgaf_proj_w')
        all_weights['mgaf_proj_b'] = tf.Variable(tf.zeros([1, self.mgaf_hidden]), name='mgaf_proj_b')

        return all_weights

    def _build_model_phase_I(self):
        self.d1_embeddings, self.d2_embeddings = self._create_bi_interaction_embed()

        mgaf_enhanced_d1, mgaf_enhanced_d2 = self._mgaf_enhance_embeddings(self.d1_embeddings, self.d2_embeddings,
                                                                           self.sparse_adj_list)

        def gate_fuse(base, mgaf, name):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                base_dim = int(base.shape[1])
                mgaf_dim = int(mgaf.shape[1])

                if mgaf_dim != base_dim:
                    mgaf = tf.layers.dense(mgaf,
                                           units=base_dim,
                                           activation=None,
                                           name='mgaf_proj')

                gate = tf.sigmoid(tf.layers.dense(tf.concat([base, mgaf], axis=1),
                                                  units=base_dim,
                                                  activation=None,
                                                  name='fuse_gate'))
                fused = gate * base + (1 - gate) * mgaf
                return fused

        self.d1_embeddings = gate_fuse(self.d1_embeddings, mgaf_enhanced_d1, name='d1_fuse')
        self.d2_embeddings = gate_fuse(self.d2_embeddings, mgaf_enhanced_d2, name='d2_fuse')

        with tf.variable_scope('mgaf_relation_map', reuse=tf.AUTO_REUSE):
            mgaf_relation_list = []
            for i in range(self.n_relations):
                adj_dense = tf.sparse_tensor_to_dense(tf.sparse.reorder(self.sparse_adj_list[i]), default_value=0.0)
                weight = tf.reduce_mean(tf.matmul(adj_dense, mgaf_enhanced_d1), axis=0, keepdims=True)
                mgaf_relation_list.append(weight)
            mgaf_global = tf.concat(mgaf_relation_list, axis=0)  # [n_relations, mgaf_hidden]

            mgaf_to_rel_w = tf.get_variable(
                name='mgaf_to_rel_w',
                shape=[self.mgaf_hidden, self.dim],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer()
            )
            mgaf_to_rel_b = tf.get_variable(
                name='mgaf_to_rel_b',
                shape=[self.dim],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            mgaf_relation_delta = tf.tanh(
                tf.matmul(mgaf_global, mgaf_to_rel_w) + mgaf_to_rel_b)  # [n_relations, dim]

            self.weights['relation_embed'] = 0.5 * self.weights['relation_embed'] + 0.5 * mgaf_relation_delta

        self.d1_e = tf.nn.embedding_lookup(self.d1_embeddings, self.drugs)
        self.pos_d2_e = tf.nn.embedding_lookup(self.d2_embeddings, self.pos_drugs)
        self.neg_d2_e = tf.nn.embedding_lookup(self.d2_embeddings, self.neg_drugs)

    def _mgaf_enhance_embeddings(self, d1_embeddings, d2_embeddings, sparse_adj_list):
        # combine adj dense sum
        adj_dense_sum = None
        for sp_adj in sparse_adj_list:
            try:
                sp_adj_reordered = tf.sparse.reorder(sp_adj)
            except Exception:
                sp_adj_reordered = sp_adj
            dense = tf.sparse_tensor_to_dense(sp_adj_reordered, default_value=0.0)
            if adj_dense_sum is None:
                adj_dense_sum = dense
            else:
                adj_dense_sum = adj_dense_sum + dense

        if adj_dense_sum is None:
            adj_dense_sum = tf.zeros([self.n_drug1, self.n_drug1], dtype=tf.float32)

        # DropEdge: randomly drop edges during training for regularization
        if self.is_training is not None and hasattr(self, 'dropedge_rate') and self.dropedge_rate > 0.0:
            def dropedge(adj_matrix, rate):
                # adj_matrix is dense [N, N]
                keep_prob = 1.0 - rate
                random_tensor = tf.random_uniform(tf.shape(adj_matrix))
                mask = tf.cast(tf.less(random_tensor, keep_prob), tf.float32)
                return adj_matrix * mask

            adj_dense_sum = tf.cond(self.is_training, lambda: dropedge(adj_dense_sum, self.dropedge_rate),
                                    lambda: adj_dense_sum)

        adj_mask = tf.cast(tf.greater(adj_dense_sum, 0.0), tf.float32)  # shape [N, N]

        # IntraGraphAttention for d1
        d1_intra = self._intra_graph_attention_tf(d1_embeddings,
                                                  adj_mask,
                                                  self.weights['mgaf_q_w'],
                                                  self.weights['mgaf_k_w'],
                                                  self.weights['mgaf_v_w'],
                                                  name_scope='d1_intra')  # [n1, mgaf_hidden]
        # IntraGraphAttention for d2
        d2_intra = self._intra_graph_attention_tf(d2_embeddings,
                                                  adj_mask,
                                                  self.weights['mgaf_q_w'],
                                                  self.weights['mgaf_k_w'],
                                                  self.weights['mgaf_v_w'],
                                                  name_scope='d2_intra')  # [n2, mgaf_hidden]

        # InterGraphAttention (d1 <-> d2)
        d1_to_d2 = self._inter_graph_attention_tf(d1_embeddings, d2_embeddings,
                                                  adj_mask,
                                                  self.weights['mgaf_inter_q_w'],
                                                  self.weights['mgaf_inter_k_w'],
                                                  self.weights['mgaf_inter_v_w'],
                                                  name_scope='d1_to_d2')  # [n1, mgaf_hidden]
        d2_to_d1 = self._inter_graph_attention_tf(d2_embeddings, d1_embeddings,
                                                  adj_mask,
                                                  self.weights['mgaf_inter_q_w'],
                                                  self.weights['mgaf_inter_k_w'],
                                                  self.weights['mgaf_inter_v_w'],
                                                  name_scope='d2_to_d1')  # [n2, mgaf_hidden]

        d1_rep = tf.concat([d1_intra, d2_to_d1], axis=1)  # [n1, mgaf_hidden*2]
        d2_rep = tf.concat([d2_intra, d1_to_d2], axis=1)  # [n2, mgaf_hidden*2]

        d1_pooled = self._sag_pool_and_broadcast(d1_rep, self.weights['mgaf_pool_score'], name_scope='d1_pool')
        d2_pooled = self._sag_pool_and_broadcast(d2_rep, self.weights['mgaf_pool_score'], name_scope='d2_pool')

        proj_w = self.weights['mgaf_proj_w']  # [mgaf_hidden*2, mgaf_hidden]
        proj_b = self.weights['mgaf_proj_b']  # [1, mgaf_hidden]

        mgaf_enhanced_d1 = tf.nn.relu(tf.matmul(d1_pooled, proj_w) + proj_b)  # [n1, mgaf_hidden]
        mgaf_enhanced_d2 = tf.nn.relu(tf.matmul(d2_pooled, proj_w) + proj_b)  # [n2, mgaf_hidden]

        mgaf_enhanced_d1 = tf.nn.l2_normalize(mgaf_enhanced_d1, axis=1)
        mgaf_enhanced_d2 = tf.nn.l2_normalize(mgaf_enhanced_d2, axis=1)

        return mgaf_enhanced_d1, mgaf_enhanced_d2

    def _intra_graph_attention_tf(self, node_features, adj_mask, q_w, k_w, v_w, name_scope='intra'):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            # single-head attention implementation (stable)
            Q = tf.matmul(node_features, q_w)  # [N, H]
            K = tf.matmul(node_features, k_w)  # [N, H]
            V = tf.matmul(node_features, v_w)  # [N, H]

            dk = tf.cast(tf.shape(K)[1], tf.float32)
            scores = tf.matmul(Q, tf.transpose(K)) / tf.sqrt(dk + 1e-8)  # [N, N]

            large_neg = -1e9
            masked_scores = tf.where(tf.equal(adj_mask, 1.0), scores, large_neg * tf.ones_like(scores))

            attn = tf.nn.softmax(masked_scores, axis=1)  # softmax over neighbors
            attn = tf.cond(self.is_training, lambda: tf.nn.dropout(attn, 1 - self.mess_dropout[0]), lambda: attn)

            out = tf.matmul(attn, V)  # [N, H]
            out = tf.nn.elu(out)

            # Residual + LayerNorm
            input_dim = node_features.get_shape().as_list()[1]
            out_dim = out.get_shape().as_list()[1]
            if out_dim is None:
                out_dim = tf.shape(out)[1]

            if input_dim is None or int(input_dim) != int(out_dim):
                proj_input = tf.layers.dense(node_features, units=out_dim, activation=None,
                                             name=name_scope + '_res_proj')
            else:
                proj_input = node_features

            residual_scale = 0.2
            fused = proj_input + residual_scale * out
            fused = tf.contrib.layers.layer_norm(fused, begin_norm_axis=1, scope=name_scope + '_layer_norm')

            fused = tf.cond(self.is_training, lambda: tf.nn.dropout(fused, 1 - self.mess_dropout[0]), lambda: fused)

            return fused

    def _inter_graph_attention_tf(self, src_features, tgt_features, adj_mask, q_w, k_w, v_w, name_scope='inter'):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            Q = tf.matmul(src_features, q_w)  # [Ns, H]
            K = tf.matmul(tgt_features, k_w)  # [Nt, H]
            V = tf.matmul(tgt_features, v_w)  # [Nt, H]

            dk = tf.cast(tf.shape(K)[1], tf.float32)
            scores = tf.matmul(Q, tf.transpose(K)) / tf.sqrt(dk + 1e-8)  # [Ns, Nt]

            large_neg = -1e9
            mask = adj_mask  # expected [Ns, Nt]
            masked_scores = tf.where(tf.equal(mask, 1.0), scores, large_neg * tf.ones_like(scores))

            attn = tf.nn.softmax(masked_scores, axis=1)  # softmax over target nodes
            attn = tf.cond(self.is_training, lambda: tf.nn.dropout(attn, 1 - self.mess_dropout[0]), lambda: attn)

            out = tf.matmul(attn, V)  # [Ns, H]
            out = tf.nn.elu(out)

            input_dim = src_features.get_shape().as_list()[1]
            out_dim = out.get_shape().as_list()[1]
            if out_dim is None:
                out_dim = tf.shape(out)[1]

            if input_dim is None or int(input_dim) != int(out_dim):
                proj_input = tf.layers.dense(src_features, units=out_dim, activation=None,
                                             name=name_scope + '_res_proj')
            else:
                proj_input = src_features

            residual_scale = 0.2
            fused = proj_input + residual_scale * out
            fused = tf.contrib.layers.layer_norm(fused, begin_norm_axis=1, scope=name_scope + '_layer_norm')
            fused = tf.cond(self.is_training, lambda: tf.nn.dropout(fused, 1 - self.mess_dropout[0]), lambda: fused)

            return fused

    def _sag_pool_and_broadcast(self, node_reps, pool_score_w, name_scope='sag_pool'):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            scores = tf.matmul(node_reps, pool_score_w)  # [N, 1]
            scores = tf.squeeze(scores, axis=1)  # [N]

            N = tf.shape(node_reps)[0]
            k = max(1, int(self.n_drug1 * 0.5))

            topk_vals, topk_idx = tf.nn.top_k(scores, k=k, sorted=False)
            topk_reps = tf.gather(node_reps, topk_idx)  # [k, D]
            pooled_vec = tf.reduce_mean(topk_reps, axis=0, keepdims=True)  # [1, D]
            pooled_broadcast = tf.tile(pooled_vec, [N, 1])  # [N, D]
            return pooled_broadcast

    def _build_model_phase_II(self):
        self.re_h_e, self.re_pos_t_e, self.re_neg_t_e, self.im_h_e, self.im_pos_t_e, self.im_neg_t_e, self.re_r_e, self.im_r_e = self._get_kg_inference_rotate(
            self.h, self.r, self.pos_t, self.neg_t)

    def _get_kg_inference_rotate(self, h, r, pos_t, neg_t):
        pi = 3.14159265358979323846
        re_embeddings = tf.concat([self.weights['re_drug1_embed'], self.weights['re_drug2_embed']], axis=0)
        re_embeddings = tf.expand_dims(re_embeddings, 1)
        im_embeddings = tf.concat([self.weights['im_drug1_embed'], self.weights['im_drug2_embed']], axis=0)
        im_embeddings = tf.expand_dims(im_embeddings, 1)

        re_h_e = tf.nn.embedding_lookup(re_embeddings, h)
        re_pos_t_e = tf.nn.embedding_lookup(re_embeddings, pos_t)
        re_neg_t_e = tf.nn.embedding_lookup(re_embeddings, neg_t)
        im_h_e = tf.nn.embedding_lookup(im_embeddings, h)
        im_pos_t_e = tf.nn.embedding_lookup(im_embeddings, pos_t)
        im_neg_t_e = tf.nn.embedding_lookup(im_embeddings, neg_t)

        re_h_e = tf.reshape(re_h_e, [-1, self.dim])
        re_pos_t_e = tf.reshape(re_pos_t_e, [-1, self.dim])
        re_neg_t_e = tf.reshape(re_neg_t_e, [-1, self.dim])
        im_h_e = tf.reshape(im_h_e, [-1, self.dim])
        im_pos_t_e = tf.reshape(im_pos_t_e, [-1, self.dim])
        im_neg_t_e = tf.reshape(im_neg_t_e, [-1, self.dim])

        relation = self.weights['re_relation_embed']
        relation = (tf.nn.l2_normalize(relation, dim=1) - 0.5) * pi

        r_e = tf.nn.embedding_lookup(relation, r)
        re_r_e = tf.cos(r_e)
        im_r_e = tf.sin(r_e)

        return re_h_e, re_pos_t_e, re_neg_t_e, im_h_e, im_pos_t_e, im_neg_t_e, re_r_e, im_r_e

    def _build_loss_phase_II(self):
        def _get_kg_score(re_h_e, re_pos_t_e, im_h_e, im_pos_t_e, re_r_e, im_r_e):
            re_score = tf.multiply(re_h_e, re_r_e) - tf.multiply(im_h_e, im_r_e)
            im_score = tf.multiply(re_h_e, im_r_e) + tf.multiply(im_h_e, re_r_e)

            re_score = re_score - re_pos_t_e
            im_score = im_score - im_pos_t_e
            kg_score = tf.concat([re_score, im_score], axis=1)
            kg_score = tf.reduce_sum(tf.square((kg_score)), 1, keep_dims=True)
            kg_score = tf.negative(kg_score)

            return kg_score

        pos_kg_score = _get_kg_score(self.re_h_e, self.re_pos_t_e, self.im_h_e, self.im_pos_t_e, self.re_r_e,
                                     self.im_r_e)
        neg_kg_score = _get_kg_score(self.re_h_e, self.re_neg_t_e, self.im_h_e, self.im_neg_t_e, self.re_r_e,
                                     self.im_r_e)

        maxi = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_kg_score - neg_kg_score - self.margin), 1e-8, 1.0))
        kg_loss = tf.negative(tf.reduce_mean(maxi))

        kg_reg_loss = tf.nn.l2_loss(self.re_h_e) + tf.nn.l2_loss(self.re_pos_t_e) + \
                      tf.nn.l2_loss(self.im_h_e) + tf.nn.l2_loss(self.im_pos_t_e) + \
                      tf.nn.l2_loss(self.re_r_e) + tf.nn.l2_loss(self.im_r_e) + \
                      tf.nn.l2_loss(self.re_neg_t_e) + tf.nn.l2_loss(self.im_neg_t_e)
        kg_reg_loss = kg_reg_loss / self.batch_size

        self.kge_loss2 = kg_loss
        self.reg_loss2 = self.regs[1] * kg_reg_loss
        self.loss2 = self.kge_loss2 + self.reg_loss2

        # Optimization process.
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

    def _build_total_loss(self):
        # base loss (classification) will be defined in _create_DNN_prediction as focal loss
        # incorporate memory regularization if exists
        self.total_loss = self.loss + self.loss2
        self.total_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss)

    def _setup_validation(self):
        """设置验证指标，使用固定阈值0.5"""
        # 获取预测概率
        self.prediction_prob = tf.nn.sigmoid(self.pred_embeddings)  # 形状: [batch_size, n_labels]

        # 固定阈值0.50.5
        self.predicted_labels = tf.cast(self.prediction_prob > self.fixed_threshold, tf.float32)
        print(f"Using fixed threshold: {self.fixed_threshold}")

        self.actual_labels = self.multi_label

        # 计算基于样本的准确率（所有标签都预测正确）
        correct_predictions = tf.reduce_all(
            tf.equal(tf.cast(self.predicted_labels, tf.int32), tf.cast(self.actual_labels, tf.int32)),
            axis=1
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def _create_DNN_prediction(self):
        # Gate-based Fusion DNN + Bilinear pooling + optional mgaf memory already applied
        regularizer = tf.nn.l2_loss(self.d1_e) + tf.nn.l2_loss(self.pos_d2_e)
        regularizer = regularizer / self.batch_size

        # pair stacking
        pair_feature = tf.stack([self.d1_e, self.pos_d2_e], axis=1)  # [B,2,D]

        # triplet attention block (same as before)
        def triplet_attention_block(inputs, scope='triplet_attention'):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                attn_drug = tf.reduce_mean(inputs, axis=2, keepdims=True)  # [B,2,1]
                attn_drug = tf.nn.sigmoid(attn_drug)
                out_drug = inputs * attn_drug

                inputs_trans = tf.transpose(inputs, perm=[0, 2, 1])  # [B,D,2]
                attn_feat = tf.reduce_mean(inputs_trans, axis=2, keepdims=True)  # [B,D,1]
                attn_feat = tf.nn.sigmoid(attn_feat)
                out_feat = inputs_trans * attn_feat
                out_feat = tf.transpose(out_feat, perm=[0, 2, 1])  # [B,2,D]

                out = (out_drug + out_feat) / 2.0
                return out

        triplet_out = triplet_attention_block(pair_feature)  # [B,2,D]

        drug1_trip = triplet_out[:, 0, :]
        drug2_trip = triplet_out[:, 1, :]

        # Gate-based fusion
        with tf.variable_scope('gate_fusion', reuse=tf.AUTO_REUSE):
            concat_feat = tf.concat([drug1_trip, drug2_trip], axis=1)
            gate = tf.sigmoid(
                tf.layers.dense(concat_feat, units=drug1_trip.shape[1], activation=None, name='gate_dense'))
            fused_feat = gate * drug1_trip + (1.0 - gate) * drug2_trip

        # Bilinear pooling
        bilinear_W = tf.get_variable('bilinear_W',
                                     shape=[fused_feat.shape[1], fused_feat.shape[1]],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bilinear_interaction = tf.reduce_sum(tf.multiply(tf.matmul(fused_feat, bilinear_W), fused_feat), axis=1,
                                             keepdims=True)

        final_feature = tf.concat([fused_feat, bilinear_interaction], axis=1)  # [B, D+1]

        # 打印调试信息
        print(f"fused_feat shape: {fused_feat.shape}")
        print(f"bilinear_interaction shape: {bilinear_interaction.shape}")
        print(f"final_feature shape: {final_feature.shape}")

        # DNN layers - 简化构建
        # 假设我们有3层：201->128, 128->64, 64->24
        pred_embeddings = final_feature

        # 第一层：201 -> 128
        pred_embeddings = tf.nn.relu(
            tf.matmul(pred_embeddings, self.weights['nn_w1']) + self.weights['nn_b1'])
        pred_embeddings = tf.cond(
            self.is_training,
            lambda: tf.nn.dropout(pred_embeddings, 1 - self.mess_dropout[0]),
            lambda: pred_embeddings
        )
        pred_embeddings = tf.nn.l2_normalize(pred_embeddings, dim=1)

        print(f"After layer 1, shape: {pred_embeddings.shape}")

        # 第二层：128 -> 64
        pred_embeddings = tf.nn.relu(
            tf.matmul(pred_embeddings, self.weights['nn_w2']) + self.weights['nn_b2'])
        pred_embeddings = tf.cond(
            self.is_training,
            lambda: tf.nn.dropout(pred_embeddings, 1 - self.mess_dropout[0]),
            lambda: pred_embeddings
        )
        pred_embeddings = tf.nn.l2_normalize(pred_embeddings, dim=1)

        print(f"After layer 2, shape: {pred_embeddings.shape}")

        # 输出层：64 -> n_labels
        pred_embeddings = tf.matmul(pred_embeddings, self.weights['nn_w3']) + self.weights['nn_b3']

        print(f"After output layer, shape: {pred_embeddings.shape}")

        self.pred_embeddings = pred_embeddings

        # === Loss: 多标签二进制交叉熵 ===
        labels = tf.cast(self.multi_label, tf.float32)
        logits = self.pred_embeddings

        # 计算二进制交叉熵损失
        bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        bce_loss = tf.reduce_mean(bce_loss)

        # 可选：添加focal loss的多标签版本
        if hasattr(self, 'focal_gamma') and hasattr(self, 'focal_alpha'):
            # 计算focal loss的多标签版本
            probs = tf.nn.sigmoid(logits)  # 使用logits计算概率
            # 对于正样本
            pos_weight = tf.pow(1.0 - probs, self.focal_gamma)
            pos_loss = labels * pos_weight * tf.log(probs + 1e-9)
            # 对于负样本
            neg_weight = tf.pow(probs, self.focal_gamma)
            neg_loss = (1.0 - labels) * neg_weight * tf.log(1.0 - probs + 1e-9)
            # 组合损失
            focal_loss = -tf.reduce_mean(self.focal_alpha * pos_loss + (1 - self.focal_alpha) * neg_loss)
            self.base_loss = focal_loss
        else:
            self.base_loss = bce_loss

        self.kge_loss = tf.constant(0.0, tf.float32)
        self.reg_loss = self.regs[0] * regularizer

        # Contrastive regularization optional (not enabled by default)
        self.contrastive_loss = tf.constant(0.0, tf.float32)

        # Combine losses; memory_reg_loss will be added into total in _build_total_loss
        self.loss = self.base_loss + self.kge_loss + self.reg_loss + self.contrastive_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _create_bi_interaction_embed5(self):
        d1_embedding = tf.concat([self.weights['re_drug1_embed'], self.weights['im_drug1_embed']], axis=1)
        d2_embedding = tf.concat([self.weights['re_drug2_embed'], self.weights['im_drug2_embed']], axis=1)

        ego_embeddings = tf.concat([d1_embedding, d2_embedding], axis=0)
        all_embeddings = [ego_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        d1_embeddings, d2_embeddings = tf.split(all_embeddings, [self.n_drug1, self.n_drug2], 0)
        return d1_embeddings, d2_embeddings

    def _create_bi_interaction_embed(self):
        pi = 3.14159265358979323846
        d1_embedding1 = tf.concat([self.weights['re_drug1_embed'], self.weights['im_drug1_embed']], axis=1)
        d2_embedding1 = tf.concat([self.weights['re_drug2_embed'], self.weights['im_drug2_embed']], axis=1)
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Data", "ddinter_label", "4", "1565_DDInter_4.csv"
            )
        )

        features = df['feature'].apply(lambda x: [float(val[:-1]) for val in x[1:-1].split(',')]).tolist()
        features_tensor = tf.constant(features, dtype=tf.float32)

        d1_embedding = tf.slice(features_tensor, [0, 0], [1565, 128])

        d2_embedding = d1_embedding
        relation_embedding = []
        for i in range(self.n_relations):
            weights = tf.reshape(self.weights['alpha'][i], [-1, 1, 1])
            relation_matrix_temp = self.weights['relation_matrix'] * weights
            relation_matrix_temp = tf.reduce_sum(relation_matrix_temp, axis=0)
            relation_embedding.append(relation_matrix_temp)

        d1_neigh, d2_neigh = [], []
        for i in range(self.n_relations):
            r_d2_embedding = d2_embedding @ relation_embedding[i]
            weight_d2_embedding = r_d2_embedding * self.weights['relation_2_att'][i]
            weight_d2_embedding = weight_d2_embedding + d2_embedding
            relation_1_neigh = tf.sparse_tensor_dense_matmul(self.sparse_adj_list[i], weight_d2_embedding)

            r_d1_embedding = d1_embedding @ relation_embedding[i]
            weight_d1_embedding = r_d1_embedding * self.weights['relation_1_att'][i]
            weight_d1_embedding = weight_d1_embedding + d1_embedding
            relation_2_neigh = tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.sparse_adj_list[i]),
                                                             weight_d1_embedding)

            d1_neigh.append(relation_1_neigh)
            d2_neigh.append(relation_2_neigh)

        d1_neigh = tf.reduce_sum(d1_neigh, 0)
        d2_neigh = tf.reduce_sum(d2_neigh, 0)
        neigh_embed = tf.concat([d1_neigh, d2_neigh], axis=0)

        ego_embeddings = tf.concat([d1_embedding1, d2_embedding1], axis=0)
        all_embeddings = []

        side_embeddings = neigh_embed

        side_embeddings = tf.concat([ego_embeddings, side_embeddings], 1)
        pre_embeddings = tf.nn.relu(
            tf.matmul(side_embeddings, self.weights['W_mlp_0']) + self.weights['b_mlp_0'])

        pre_embeddings = tf.cond(
            self.is_training,
            lambda: tf.nn.dropout(pre_embeddings, 1 - self.mess_dropout[0]),
            lambda: pre_embeddings
        )

        all_embeddings += [pre_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        d1_embeddings, d2_embeddings = tf.split(all_embeddings, [self.n_drug1, self.n_drug2], 0)
        return d1_embeddings, d2_embeddings

    def _statistics_params(self):
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        feed_dict[self.is_training] = True
        return sess.run([self.total_opt, self.loss, self.base_loss, self.kge_loss, self.reg_loss, \
                         self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

    def eval(self, sess, feed_dict):
        """评估函数，使用固定阈值0.5"""
        feed_dict[self.is_training] = False
        batch_predictions, batch_labels = sess.run([self.prediction_prob, self.predicted_labels], feed_dict)
        return batch_predictions, batch_labels

    def eval_with_metrics(self, sess, feed_dict):
        """评估函数，使用固定阈值0.5，返回预测概率和准确率"""
        feed_dict[self.is_training] = False
        predictions, predicted_labels, accuracy = sess.run([self.prediction_prob, self.predicted_labels, self.accuracy],
                                                           feed_dict)
        return predictions, predicted_labels, accuracy