import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.compat.v1.nn.rnn_cell import LSTMCell

import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score
import random
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import os
import time
import datetime
import signal
import math
import pickle

TRAING_TIME = 15
SHUFFLE = True
LOAD_LITTLE_DATA = False

class SparseData():

    def shuffle(self):
        if SHUFFLE:
            np.random.shuffle(self.index)
        return self.data[self.index], self.seqlen[self.index], self.labels[self.index], self.sublabels[self.index], self.subsublabels[self.index]

    def __init__(self, surv, INPUT_FILE, MAX_SEQ_LEN):
        """
        surv=1 <-> WIN <-> DEAD
        """
        self.data = []
        labels = []
        self.seqlen = []
        self.sublabels = []
        self.subsublabels = []
        self.SURV_DIST = np.ones(MAX_SEQ_LEN)
        cods = []
        ids = []

        df = pd.read_csv("../died_curve.csv")
        curve = df['x'].to_numpy()
        assert curve.shape[0] == MAX_SEQ_LEN
        self.DEATH_DIST = curve

        # read data, select columns
        print("*******", INPUT_FILE)
        df = pd.read_csv(INPUT_FILE)
        print("\tfeature size is {}".format(len(df.columns) - 2))

        # dataframe to data
        for idx, row in df.iterrows():
            if row['status'] == surv:
                x = df.iloc[idx].to_numpy()[:-2]
                time = int(row['time'])
                status = int(row['status'])

                self.data.append(x)
                labels.append(status)
                ids.append(idx)

                if status == 1:
                    self.sublabels.append(self.DEATH_DIST)
                    self.seqlen.append(time)
                    sub_ = np.ones(MAX_SEQ_LEN)
                    sub_[time:] = 0
                    self.subsublabels.append(sub_)
                else:
                    self.sublabels.append(self.SURV_DIST)
                    self.seqlen.append(MAX_SEQ_LEN)
                    self.subsublabels.append(np.ones(MAX_SEQ_LEN))

        self.size = len(self.data)
        self.data = np.array(self.data)
        labels = np.array(labels)
        b = np.zeros((labels.size, 2))
        b[np.arange(labels.size), labels] = 1
        self.labels = b
        self.seqlen = np.array(self.seqlen)
        self.sublabels = np.array(self.sublabels)
        self.subsublabels = np.array(self.subsublabels)
        print("data size ", self.size, "\n")
        # self.index = -range(0, self.size)
        self.index = np.arange(self.size)
        self.data, self.seqlen, self.labels, self.sublabels, self.subsublabels = self.shuffle()
        self.batch_id = 0


    def next(self, batch_size):
        if self.batch_id + batch_size > len(self.data):
            self.data, self.seqlen, self.labels, self.sublabels, self.subsublabels = self.shuffle()
            self.batch_id = 0
            self.finish_epoch = True
        batch_data = self.data[self.batch_id:self.batch_id + batch_size]
        batch_labels = self.labels[self.batch_id:self.batch_id + batch_size]
        batch_seqlen = self.seqlen[self.batch_id:self.batch_id + batch_size]
        batch_sublabels = self.sublabels[self.batch_id:self.batch_id + batch_size]
        batch_subsublabels = self.subsublabels[self.batch_id:self.batch_id + batch_size]
        self.batch_id = self.batch_id + batch_size
        return np.array(batch_data), np.array(batch_labels), np.array(batch_seqlen), np.array(batch_sublabels), np.array(batch_subsublabels)

    def next_half_batch(self, batch_size):
        batch_size = int(batch_size/2)
        if self.batch_id + batch_size > len(self.data):
            self.data, self.seqlen, self.labels, self.sublabels, self.subsublabels = self.shuffle()
            self.batch_id = 0
            self.finish_epoch = True
        batch_data = self.data[self.batch_id:self.batch_id + batch_size]
        batch_labels = self.labels[self.batch_id:self.batch_id + batch_size]
        batch_seqlen = self.seqlen[self.batch_id:self.batch_id + batch_size]
        batch_sublabels = self.sublabels[self.batch_id:self.batch_id + batch_size]
        batch_subsublabels = self.subsublabels[self.batch_id:self.batch_id + batch_size]
        self.batch_id = self.batch_id + batch_size
        return np.array(batch_data), np.array(batch_labels), np.array(batch_seqlen), np.array(batch_sublabels), np.array(batch_subsublabels)

class biSparseData():
    def __init__(self, INPUT_FILE, MAX_SEQ_LEN):
        random.seed(time.time())
        self.winData = SparseData(1, INPUT_FILE, MAX_SEQ_LEN)
        self.loseData = SparseData(0, INPUT_FILE, MAX_SEQ_LEN)
        self.size = self.winData.size + self.loseData.size
    def next_balance(self, batch):
        a, b, c, d, e = self.winData.next_half_batch(batch)
        f, g, h, i, j = self.loseData.next_half_batch(batch)
        return np.concatenate((a, f), axis=0), np.concatenate((b, g), axis=0), np.concatenate((c, h), axis=0), np.concatenate((d, i), axis=0), np.concatenate((e, j), axis=0)


class BASE_RNN():

    train_data = None
    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def __init__(self,  EMB_DIM = 32,
                        FEATURE_SIZE = 108,
                        BATCH_SIZE = 128,
                        MAX_SEQ_LEN = 350,
                        TRAING_STEPS = 100000,
                        STATE_SIZE = 64,
                        LR = 0.001,
                        LRD = 0.99,
                        GRAD_CLIP = 5.0,
                        L2_NORM = 0.001,
                        INPUT_FILE = "2997",
                        ADD_TIME_FEATURE=False,
                        MIDDLE_FEATURE_SIZE = 30,
                        LOG_FILE_NAME=None,
                        FIND_PARAMETER = False,
                        SAVE_LOG=True,
                        OPEN_TEST=True,
                        LOG_PREFIX="",
                        TEST_FREQUENT=False,
                        DNN_MODEL = False,
                        QRNN_MODEL = False,
                        GLOAL_STEP = 0,
                        COV_SIZE = 1,
                        DOUBLE_QRNN = False,
                        LOSS_FUNC = "ce"
):
        self.DOUBLE_QRNN = DOUBLE_QRNN
        self.QRNN_MODEL = QRNN_MODEL
        self.global_step = GLOAL_STEP
        self.DNN_MODEL = DNN_MODEL
        self.TEST_FREQUENT = TEST_FREQUENT
        self.FIND_PARAMETER = FIND_PARAMETER
        self.add_time_feature = ADD_TIME_FEATURE
        self.MIDDLE_FEATURE_SIZE = MIDDLE_FEATURE_SIZE
        tf.reset_default_graph()
        self.TRAING_STEPS = TRAING_STEPS
        self.BATCH_SIZE = BATCH_SIZE
        self.STATE_SIZE = STATE_SIZE
        self.EMB_DIM = EMB_DIM
        self.FEATURE_SIZE = FEATURE_SIZE
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.LR = LR
        self.LRD = LRD
        self.GRAD_CLIP = GRAD_CLIP
        self.L2_NORM = L2_NORM
        self.INPUT_FILE = INPUT_FILE
        self.SAVE_LOG = SAVE_LOG
        # self.TRAIN_FILE = "../data/" + INPUT_FILE + "/train.yzbx.txt"
        # self.TEST_FILE = "../data/" + INPUT_FILE + "/test.yzbx.txt"
        self.TRAIN_FILE = "../data/" + INPUT_FILE + "/seer_train.csv"
        self.TEST_FILE = "../data/" + INPUT_FILE + "/seer_test.csv"
        self.OPEN_TEST = OPEN_TEST
        self.COV_SIZE = COV_SIZE
        self.LOSS_FUNC = LOSS_FUNC

  
        para = None
        if LOG_FILE_NAME != None:
            para = LOG_FILE_NAME
        else:
            para = LOG_PREFIX + str(self.EMB_DIM) + "_" + \
                str(BATCH_SIZE) + "_" + \
                str(self.STATE_SIZE) + "_" + \
                "{:.6f}".format(self.LR) + "_" + \
                "{:.6f}".format(self.L2_NORM) + "_" + \
                INPUT_FILE + "_" + str(ADD_TIME_FEATURE) + \
                   "_" + str(self.QRNN_MODEL) + "_" + str(self.COV_SIZE)
        print(para, '\n')
        self.filename = para
        self.train_log_txt_filename = "./" + para + '.train.log.txt'
        if os.path.exists(self.train_log_txt_filename):
            self.exist = True
        else:
            if self.SAVE_LOG:
                self.exist = False
                self.train_log_txt = open(self.train_log_txt_filename, 'w')
                self.train_log_txt.close()

    def get_survival_data(self, model, sess):
        alltestdata = SparseData(self.TEST_FILE, True, True)
        ret = []
        while alltestdata.finish_epoch == False:
            test_batch_x, test_batch_y, test_batch_len = alltestdata.next(self.BATCH_SIZE)
            bid_loss, bid_test_prob, preds = sess.run(
                [self.cost, self.predict, self.preds],
                feed_dict={self.tf_x: test_batch_x,
                           self.tf_y: test_batch_y,
                           self.tf_bid_len: test_batch_len,
                           })
            ret.append(preds)
        return ret


    def load_data(self):
        self.train_data = biSparseData(self.TRAIN_FILE, self.MAX_SEQ_LEN)
        self.test_data_win = SparseData(1, self.TEST_FILE, self.MAX_SEQ_LEN)
        self.test_data_lose = SparseData(0, self.TEST_FILE, self.MAX_SEQ_LEN)

    def is_exist(self):
        if self.SAVE_LOG == False:
            return False
        return self.exist


    def create_graph(self):
        BATCH_SIZE = self.BATCH_SIZE
        self.tf_x = tf.placeholder(tf.float32, [BATCH_SIZE, self.FEATURE_SIZE], name="tf_x")
        self.tf_y = tf.placeholder(tf.float32, [BATCH_SIZE, 2], name="tf_y")
        self.tf_sub_y = tf.placeholder(tf.float32, [BATCH_SIZE, self.MAX_SEQ_LEN], name="tf_sub_y")
        self.tf_subsub_y = tf.placeholder(tf.float32, [BATCH_SIZE, self.MAX_SEQ_LEN], name="tf_subsub_y")
        self.tf_bid_len = tf.placeholder(tf.int32, [BATCH_SIZE], name="tf_len")
        self.tf_rnn_len = tf.ones(BATCH_SIZE) * self.MAX_SEQ_LEN
        # self.tf_rnn_len = self.tf_bid_len + 2
        input_x = None

        self.w0 = None  # embedding_x w
        self.w1 = None  # 
        if self.add_time_feature:
            middle_layer = tf.layers.dense(self.tf_x, self.MIDDLE_FEATURE_SIZE, tf.nn.relu, name="embedding_x")  # hidden layer
            with tf.variable_scope('embedding_x', reuse=True):
                w = tf.get_variable('kernel')
                self.w0 = w
            self.middle_layer = middle_layer
            
            def add_time(x):
                y = tf.reshape(tf.tile(x, [self.MAX_SEQ_LEN]), [self.MAX_SEQ_LEN, self.MIDDLE_FEATURE_SIZE]) # (239, feature_size)
                t = tf.reshape(tf.range(self.MAX_SEQ_LEN), [self.MAX_SEQ_LEN, 1])  # (239, 31)
                z = tf.concat([y, tf.cast(t, dtype=tf.float32)], 1)  # (239, 1)
                
                # print("shape of input=", y.get_shape().as_list(), z.get_shape().as_list(), t.get_shape().as_list())  
                return z

            input_x = tf.map_fn(add_time, middle_layer)
            # print("shape of input=", input_x.get_shape().as_list())  # (128, 239, 31)

        self.input_x = input_x
        preds = None

        if self.DNN_MODEL:
            # outlist = []
            # for i in range(0, self.BATCH_SIZE):
            #     single = tf.layers.dense(input_x[i], 1, tf.nn.sigmoid, name="embedding_timestamp_{}".format(i))  # (239, 31) -> (239, 1)
            #     single = tf.reshape(single, [1, self.MAX_SEQ_LEN])
            #     outlist.append(single)
            
            #     with tf.variable_scope('embedding_timestamp_{}'.format(i), reuse=True):
            #         w = tf.get_variable('kernel')
            #         self.w1 = w
            #         print("shape of w=", w.get_shape().as_list())  # (128, 239, 31)
            # # self.w1 = tf.get_default_graph().get_tensor_by_name(os.path.split(sigleout.name)[0] + '/kernal:0')
            # preds = tf.reshape(tf.stack(outlist, axis=0), [self.BATCH_SIZE, self.MAX_SEQ_LEN], name="preds")  # (128, 239)

            with tf.variable_scope('dnn'):
                WW = tf.get_variable('WW', [self.MAX_SEQ_LEN, self.MIDDLE_FEATURE_SIZE + 1, 1])
                # b = tf.get_variable('b', [self.STATE_SIZE, 1], initializer=tf.constant_initializer(0))

            self.w1 = WW
            predss = tf.einsum("bml, mlk -> bmk", input_x, WW)
            predsss = tf.reshape(predss, [BATCH_SIZE, self.MAX_SEQ_LEN])
            preds = tf.math.sigmoid(predsss)
        else:
            # input_x = tf.reshape(tf.tile(input, [1, self.MAX_SEQ_LEN]), [BATCH_SIZE, self.MAX_SEQ_LEN, self.FEATURE_SIZE * self.EMB_DIM])
            rnn_cell = None
            rnn_cell = LSTMCell(num_units=self.STATE_SIZE)


            outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
                rnn_cell,                   # cell you have chosen
                input_x,                    # input
                initial_state=None,         # the initial hidden state
                dtype=tf.float32,           # must given if set initial_state = None
                time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
                sequence_length=self.tf_rnn_len
            )

            new_output = tf.reshape(outputs, [self.MAX_SEQ_LEN * BATCH_SIZE, self.STATE_SIZE])  # new_output.shape = (30592, 128)

            with tf.variable_scope('softmax'):
                W = tf.get_variable('W', [self.STATE_SIZE, 1])
                b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0))

            logits = tf.matmul(new_output, W) + b  # (30592, 1)
            preds = tf.transpose(tf.nn.sigmoid(logits, name="preds"), name="preds")[0]  # (30592, 1)
        
        self.preds = preds
        survival_rate = preds

        self.batch_rnn_survival_rate = tf.reshape(survival_rate, [BATCH_SIZE, self.MAX_SEQ_LEN])  # (128, 239)
        condition1 = self.tf_subsub_y > 0.5
        condition0 = self.tf_subsub_y < 0.5
        fill1 = tf.fill([BATCH_SIZE, self.MAX_SEQ_LEN],2.0)
        fill0 = tf.fill([BATCH_SIZE, self.MAX_SEQ_LEN],-2.0)
        s1 = tf.where(condition=condition1,x=self.batch_rnn_survival_rate,y=fill1)
        s0 = tf.where(condition=condition0,x=self.batch_rnn_survival_rate,y=fill0)
        minimum1 = tf.reduce_min(s1,0)
        maximum0 = tf.reduce_max(s0,0)
        com = tf.zeros(self.MAX_SEQ_LEN)
        self.nn = tf.reduce_sum(tf.math.square(tf.maximum(com, tf.subtract(maximum0, minimum1)))) * 10
        
        died_p = tf.math.multiply(
                    self.batch_rnn_survival_rate, 
                    tf.tile(tf.reshape(self.tf_y[:, 1], [BATCH_SIZE, 1]), tf.constant([1, self.MAX_SEQ_LEN], tf.int32)))
        alive_p = tf.math.multiply(
                    self.batch_rnn_survival_rate, 
                    tf.tile(tf.reshape(self.tf_y[:, 0], [BATCH_SIZE, 1]), tf.constant([1, self.MAX_SEQ_LEN], tf.int32)))
        max_died = tf.reduce_max(died_p,0)           
        condition1 = alive_p > 0.0
        fill1 = tf.fill([BATCH_SIZE, self.MAX_SEQ_LEN],2.0)
        s1 = tf.where(condition=condition1,x=alive_p,y=fill1) 
        min_alive = tf.reduce_min(s1,0)
        com = tf.zeros(self.MAX_SEQ_LEN)
        self.nn1 = tf.reduce_sum(tf.math.square(tf.maximum(com, tf.subtract(max_died, min_alive)))) * 100
        
        sk = self.batch_rnn_survival_rate[:, 0: self.MAX_SEQ_LEN - 1]
        sk1 = self.batch_rnn_survival_rate[:, 1: self.MAX_SEQ_LEN]
        com = tf.zeros((BATCH_SIZE, self.MAX_SEQ_LEN-1))
        self.mm = tf.reduce_sum(tf.math.square(tf.maximum(com, tf.subtract(sk1, sk)))) * 10000
        
        area = tf.cast(tf.add(self.batch_rnn_survival_rate[:,0],self.batch_rnn_survival_rate[:,self.MAX_SEQ_LEN - 1]),dtype = tf.float32)/2 + tf.cast(tf.reduce_sum(self.batch_rnn_survival_rate[:, 1:self.MAX_SEQ_LEN],1),dtype = tf.float32) # 128
        mse = tf.keras.losses.MeanSquaredError()
        self.surv_time = mse(area,tf.cast(self.tf_bid_len,dtype = tf.float32))
        
        self.preds_status = tf.concat([
            self.batch_rnn_survival_rate,
            tf.cast(tf.reshape(self.tf_bid_len, [BATCH_SIZE, 1]), tf.float32)], 1)
        self.cross_entropy = -tf.reduce_sum(
            self.tf_subsub_y*tf.log(tf.clip_by_value(self.batch_rnn_survival_rate,1e-10,1.0)) + 
            tf.subtract(tf.constant(1, tf.float32), self.tf_subsub_y) * tf.log(tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.clip_by_value(self.batch_rnn_survival_rate,1e-10,1.0)))
            ) / (self.BATCH_SIZE * self.MAX_SEQ_LEN)
        
        scalar = 2 / self.BATCH_SIZE
        batch_win_mean_survival_rate = tf.multiply(
            scalar, 
            tf.reduce_sum(
                tf.math.multiply(
                    self.batch_rnn_survival_rate, 
                    tf.tile(tf.reshape(self.tf_y[:, 1], [BATCH_SIZE, 1]), tf.constant([1, self.MAX_SEQ_LEN], tf.int32))), 0))
        self.batch_win_mean_survival_rate = batch_win_mean_survival_rate
        batch_lose_mean_survival_rate = tf.multiply(
            scalar, 
            tf.reduce_sum(
                tf.math.multiply(
                    self.batch_rnn_survival_rate, 
                    tf.tile(tf.reshape(self.tf_y[:, 0], [BATCH_SIZE, 1]), tf.constant([1, self.MAX_SEQ_LEN], tf.int32))), 0))
        scalar = 2.
        batch_win_true = tf.multiply(
            scalar, tf.reduce_mean(tf.math.multiply(self.tf_sub_y, tf.tile(tf.reshape(self.tf_y[:, 1], [BATCH_SIZE, 1]), tf.constant([1, self.MAX_SEQ_LEN], tf.int32))), 0))
        batch_lose_true = tf.multiply(
            scalar, tf.reduce_mean(tf.math.multiply(self.tf_sub_y, tf.tile(tf.reshape(self.tf_y[:, 0], [BATCH_SIZE, 1]), tf.constant([1, self.MAX_SEQ_LEN], tf.int32)), 0)))

        map_parameter = tf.concat([self.batch_rnn_survival_rate,
                                   tf.cast(tf.reshape(self.tf_bid_len, [BATCH_SIZE, 1]), tf.float32)],
                                  1)

        def reduce_mul(x):
            bid_len = tf.cast(x[self.MAX_SEQ_LEN - 1], dtype=tf.int32)
            survival_rate_last_one = x[bid_len]
            return survival_rate_last_one
            

        # self.final_survival_rate = tf.map_fn(reduce_mul, elems=map_parameter, name="rate_result")
        self.final_survival_rate = self.batch_rnn_survival_rate[:, -1]

        mse = tf.keras.losses.MeanSquaredError()
        self.mse_cost_win = mse(batch_win_mean_survival_rate, batch_win_true)
        self.mse_cost_lose = mse(batch_lose_mean_survival_rate, batch_lose_true)
        tvars = tf.trainable_variables()
        self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars ]) * self.L2_NORM / self.BATCH_SIZE
        self.mse_cost = tf.add(self.mse_cost_win, self.mse_cost_lose)
        
        self.cross_entropy_last = -tf.reduce_sum(
            self.tf_y[:, 0]*tf.log(tf.clip_by_value(self.batch_rnn_survival_rate[:, -1],1e-10,1.0)) + 
            tf.subtract(tf.constant(1, tf.float32), self.tf_y[:, 0]) * tf.log(tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.clip_by_value(self.batch_rnn_survival_rate[:, -1],1e-10,1.0)))
            ) / self.BATCH_SIZE
        

        cost = tf.constant(0.0, dtype=tf.float32)
        names = self.LOSS_FUNC.split("+")
        for name in names:
            if name == "ce":
                cost = tf.add(self.cross_entropy, cost)
            elif name == "mse":
                cost = tf.add(self.mse_cost, cost)
            elif name == "mm":
                cost = tf.add(self.mm, cost)
            elif name == "nn":
                cost = tf.add(self.nn, cost)
            elif name == "nn1":
                cost = tf.add(self.nn1, cost)
            elif name == "time":
                cost = tf.add(self.surv_time, cost)
            else:
                print("WRONG LOSS FUNCTION!")
                return
        self.cost = tf.add(cost, self.lossL2, name="cost")


        final_dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), self.final_survival_rate)
        self.predict = tf.transpose(tf.stack([self.final_survival_rate, final_dead_rate]), name="predict")


        # optimize
        learning_rate = tf.train.exponential_decay(learning_rate=self.LR, global_step=self.global_step, decay_steps=10, decay_rate=self.LRD, staircase=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=0.99)#.minimize(cost)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.GRAD_CLIP)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), name="train_op")
        tf.add_to_collection('train_op', self.train_op)

        correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.tf_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")


    def train_test(self,sess):
        self.load_data()
        init = tf.global_variables_initializer()
        self.sess = sess
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=100)
        self.saver = saver
        TRAIN_LOG_STEP = int((self.train_data.size * 0.1) / self.BATCH_SIZE)
        train_auc_arr = []
        train_loss_arr = []
        train_auc_label = []
        train_auc_prob = []
        total_train_duration = 0
        total_test_duration = 0
        TEST_COUNT = 0
        max_auc = -1
        enough_test = 0
        test_min_cost = 100000
        last_loss = [9999.0, 9999.0]
        start_time = time.time()
        for step in range(1, self.TRAING_STEPS + 1):
            self.global_step = step
            batch_x, batch_y, batch_len, batch_sub_y, batch_subsub_y = self.train_data.next_balance(self.BATCH_SIZE)
            
            _, middlelayer, predd, inputx, inputx_dense0, inputx_dense1, train_loss, train_outputs, ce, l2, mse, mm, nn, preds_st = sess.run([self.train_op, self.middle_layer, self.preds, self.input_x, self.w0, self.w1, self.cost, self.predict, self.cross_entropy, self.lossL2, self.mse_cost, self.mm, self.nn1, self.preds_status], feed_dict={self.tf_x: batch_x,
                                                self.tf_y: batch_y,
                                                self.tf_bid_len: batch_len,
                                                self.tf_sub_y: batch_sub_y,
                                                self.tf_subsub_y: batch_subsub_y
                                                })
            
            #print train_outputs
            # print("xxxxxxtf {} \n\n step_{} \n\t preds {} \n\n inputx{} \n\n middlelayer {} \n\n dense w0{} \n\n dense w1{}".format(batch_x,  step, predd, inputx, middlelayer, inputx_dense0, inputx_dense1))
            # print("\t cross entropy {}, l2 {}, mse {}, mm {}, nn1 {}".format(ce, l2, mse, mm, nn))
            train_loss_arr.append(train_loss)
            train_auc_label.append(batch_y.T[0])
            train_auc_prob.append(np.array(train_outputs).T[0])

            # logging
            mean_loss = np.array(train_loss_arr[-99:]).mean()
            mean_auc = 0.0001
            try:
                mean_auc = roc_auc_score(np.reshape(train_auc_label, [1, -1])[0], np.reshape(train_auc_prob, [1, -1])[0])
            except Exception:
                print("AUC ERROR")
                continue

            if self.global_step % 100 == 0:
                test_win_loss, test_win_acc, test_lose_loss, test_lose_acc, data = self.run_test(sess)
                with open("log_bc_{}_dnn{}.txt".format(self.LOSS_FUNC, self.DNN_MODEL), "a") as text_file:
                    text_file.write("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(self.global_step, mean_loss, mean_auc, (test_win_loss + test_lose_loss)/2., (test_win_acc + test_lose_acc)/2.))
                with open('../output/output_bc_{}_{}.pickle'.format(self.LOSS_FUNC, self.global_step), 'wb') as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                log = self.getStatStr("TRAIN", self.global_step, mean_auc, mean_loss)
                print(log)
                self.force_write(log)

    def run_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.train_test(sess)

    def save_model(self):
        print("model name: ", self.filename, " ", self.global_step, "\n")
        self.saver.save(self.sess, "./saved_model/model" + self.filename, global_step=self.global_step)

    def getStatStr(self, category ,step, mean_auc, mean_loss):
        statistics_log = str(self.INPUT_FILE) + "\t" + category + "\t" + str(step) + "\t" \
                         "{:.6f}".format(mean_loss) + "\t" + \
                         "{:.4f}".format(mean_auc) + "\t" + \
                         str(self.EMB_DIM) + "\t" + str(self.BATCH_SIZE) + "\t" + \
                         str(self.STATE_SIZE) + "\t" + \
                         "{:.6f}".format(self.LR) + "\t" + \
                         "{:.6}".format(self.L2_NORM) + "\n"
        return statistics_log

    def load(self, meta, ckpt, step):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.import_meta_graph(meta)
        #self.load_data()
        self.global_step = step
        #with tf.Session(config=config) as sess:
        sess = tf.Session(config=config)
        saver.restore(sess, ckpt)
        graph = tf.get_default_graph()
        self.tf_x = graph.get_tensor_by_name("tf_x:0")
        self.tf_y = graph.get_tensor_by_name("tf_y:0")
        self.tf_sub_y = graph.get_tensor_by_name("tf_sub_y:0")
        self.tf_subsub_y = graph.get_tensor_by_name("tf_subsub_y:0")
        self.tf_bid_len = graph.get_tensor_by_name("tf_len:0")
        self.tf_market_price = graph.get_tensor_by_name("tf_market_price:0")
        self.accuracy = graph.get_tensor_by_name("accuracy:0")
        self.cost = graph.get_tensor_by_name("cost:0")
        self.predict = graph.get_tensor_by_name("predict:0")
        self.train_op = tf.get_collection('train_op')[0]

        #self.anlp_train_op = graph.get_collection("anlp_train_op")[0]
        #self.train _op = graph.get_tensor_by_name("train_op:0")
        self.preds = graph.get_tensor_by_name("preds:0")
        #self.com_train_op = tf.get_collection("com_train_op")[0]
        #self.tf_control_parameter = graph.get_tensor_by_name("tf_control_parameter:0")
        # self.train_log_txt.write(statistics_log)
        return sess

    def run_test(self, sess):
        auc_arr = []
        loss_arr = []
        auc_prob = []
        auc_label = []
        h_win_pred = None
        h_lose_pred = None

        # print self.test_data_win.size + self.test_data_lose.size, "total size"
        # print("TEST WIN DATA size {}".format(self.test_data_win.size))
        total_time = 0
        for i in range(0, int(self.test_data_win.size / self.BATCH_SIZE)):
            test_batch_x, test_batch_y, test_batch_len, test_batch_sub_y, test_batch_subsub_y = self.test_data_win.next(
                self.BATCH_SIZE)
            start_time = time.time()
            bid_loss, batch_win_acc, win_preds = sess.run(
                [self.cost, self.accuracy, self.preds_status],
                feed_dict={self.tf_x: test_batch_x,
                           self.tf_y: test_batch_y,
                           self.tf_bid_len: test_batch_len,
                           self.tf_sub_y: test_batch_sub_y,
                           self.tf_subsub_y: test_batch_subsub_y
                           })
            total_time += time.time() - start_time
            loss_arr.append(bid_loss)
            

            if h_win_pred is None:
                h_win_pred = np.array(win_preds)
            else:
                h_win_pred = np.concatenate((h_win_pred, np.array(win_preds)), axis=0)

            auc_arr.append(batch_win_acc)
        win_mean_auc = np.array(auc_arr).mean()
        win_mean_loss = np.array(loss_arr).mean()

        log = self.getStatStr("TEST_WIN_DATA", self.global_step, win_mean_auc, win_mean_loss)
        # print(log)

        # print("ADD TEST LOSE DATA size {}".format(self.test_data_lose.size))
        auc_arr = []
        for i in range(0, int(self.test_data_lose.size / self.BATCH_SIZE)):
            test_batch_x, test_batch_y, test_batch_len, test_batch_sub_y, test_batch_subsub_y = self.test_data_lose.next(
                self.BATCH_SIZE)
            bid_loss, batch_lose_acc, lose_preds = sess.run(
                                   [self.cost, self.accuracy, self.preds_status],
                                   feed_dict={self.tf_x: test_batch_x,
                                              self.tf_y: test_batch_y,
                                              self.tf_bid_len: test_batch_len,
                                              self.tf_sub_y: test_batch_sub_y,
                                              self.tf_subsub_y: test_batch_subsub_y
                                              })
            loss_arr.append(bid_loss)

            if h_lose_pred is None:
                h_lose_pred = np.array(lose_preds)
            else:
                h_lose_pred = np.concatenate((h_lose_pred, np.array(lose_preds)), axis=0)

            auc_arr.append(batch_lose_acc)
        lose_mean_loss = np.array(loss_arr).mean()
        lose_mean_auc = np.array(auc_arr).mean()
        log = self.getStatStr("TEST", self.global_step, lose_mean_auc, lose_mean_loss)
        self.force_write(log)
        # print(log)

        
        
        # save preds
        assert win_preds.shape[1] == self.MAX_SEQ_LEN + 1
        assert lose_preds.shape[1] == self.MAX_SEQ_LEN + 1
        data = h_win_pred, h_lose_pred

        return win_mean_loss, win_mean_auc, lose_mean_loss, lose_mean_auc, data
        # with open('../output/output_{}_trainloss_{}_trainacc_{}_winloss_{}_winacc_{}_loseloss_{}_loseacc_{}.pickle'.format(self.global_step, train_loss, train_acc, win_mean_loss, win_mean_auc, lose_mean_loss, lose_mean_auc), 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

    def force_write(self, log):
        if not self.SAVE_LOG:
            return
        self.train_log_txt = open(self.train_log_txt_filename, 'a')
        self.train_log_txt.write(log)
        self.train_log_txt.close()