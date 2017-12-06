import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import random
import os
from sklearn.metrics import accuracy_score
import argparse

class linear_classifier_zsl(object):
    def __init__(self, validation, generalized, features_test, labels_test, unique_labels_test, unique_labels_test_unseen=[], unique_labels_test_seen=[], learning_rate=0.0005, number_epoch=25, batch_size=100):
        self.features_test = features_test
        self.labels_test = labels_test

        self.validation = validation
        self.generalized = generalized
        self.unique_labels_test = unique_labels_test
        self.decay_factor = 0.9
        if not self.validation:
            self.unique_labels_test_unseen = unique_labels_test_unseen
            if self.generalized:
                self.unique_labels_test_seen = unique_labels_test_seen
        if not self.generalized:
            self.class_idx = np.where(np.sum(self.labels_test, axis=0) != 0)[0]
            self.labels_test = self.labels_test[:, self.class_idx]
            self.unique_labels_test = self.unique_labels_test[:, self.class_idx]
            if not self.validation:
                self.unique_labels_test_unseen = unique_labels_test_unseen[:, self.class_idx]

        self.features_train = None
        self.labels_train = None
        self.learning_rate = learning_rate
        self.number_epoch = number_epoch
        self.batch_size = batch_size
        self.features_pl = tf.placeholder(tf.float32, shape=(None, self.features_test.shape[1]))
        self.labels_pl = tf.placeholder(tf.float32, shape=(None, self.labels_test.shape[1]))
        self.batch_size_pl = tf.placeholder(tf.int32)
        self.lr_pl = tf.placeholder(tf.float32, shape=(None))
        self.model()

    def model(self):
        self.logits_op, self.b = self.linear(self.features_pl, self.labels_test.shape[1], name="classifier")
        self.softmax_op = tf.nn.softmax(self.logits_op)
        self.loss_op = self.loss(self.logits_op, self.labels_pl)
        self.train_op = self.training(self.loss_op, self.lr_pl)

    def linear(self, input, output_dim, name=None, stddev=0.02):
        with tf.variable_scope(name or 'linear'):
            norm = tf.random_normal_initializer(stddev=stddev)
            const = tf.constant_initializer(0.0)
            w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
            b = tf.get_variable('b', [output_dim], initializer=const)
            return tf.matmul(input, w) + b, b

    def loss(self, logits, labels_pl):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_pl, logits=logits, name='softmax'))
        return loss

    def training(self, loss_func, learning_rate):
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(loss_func, global_step=global_step)

    def next_batch(self, start, end):
        if start == 0:
            idx = np.r_[:self.features_train.shape[0]]
            random.shuffle(idx)
            self.features_train = self.features_train[idx]
            self.labels_train = self.labels_train[idx]
        if end > self.features_train.shape[0]:
            end = self.features_train.shape[0]
        return self.features_train[start:end], self.labels_train[start:end]

    def compute_mean_class_accuracy(self, logits, unique_label):
        acc = 0.0
        for lab in unique_label:
            idx = np.where(np.all(self.labels_test == lab, axis=1))[0]
            acc += accuracy_score(np.argmax(self.labels_test[idx], axis=1), np.argmax(logits[idx], axis=1))
        return acc / unique_label.shape[0]

    def val(self):
        logits = self.linear_sess.run(self.softmax_op, feed_dict={self.features_pl: self.features_test,
                                                                  self.labels_pl: self.labels_test})
        acc_seen = acc_unseen = 0.0
        if self.validation:
            acc = self.compute_mean_class_accuracy(logits, self.unique_labels_test)
        else:
            acc_unseen = self.compute_mean_class_accuracy(logits, self.unique_labels_test_unseen)
            if self.generalized:
                acc_seen = self.compute_mean_class_accuracy(logits, self.unique_labels_test_seen)
                acc = (2 * (acc_seen * acc_unseen)) / (acc_seen + acc_unseen)
            else:
                acc = acc_unseen
        return acc, acc_seen, acc_unseen

    def learn(self, sess, features_train, labels_train):
        self.features_train = features_train
        self.labels_train = labels_train
        if not self.generalized:
            self.labels_train = labels_train[:, self.class_idx]
        self.features_test_temp = self.features_test
        self.linear_sess = sess
        init = tf.global_variables_initializer()
        self.linear_sess.run(init)
        self.learning_rate = 0.001
        best_acc = best_acc_seen = best_acc_unseen = 0.0
        last_loss_epoch = None
        for i in xrange(self.number_epoch):
            mean_loss_d = 0.0
            for count in xrange(0, self.features_train.shape[0], self.batch_size):
                features_batch, labels_batch = self.next_batch(count, count+self.batch_size)
                _, loss_value = self.linear_sess.run([self.train_op, self.loss_op],
                                                     feed_dict={self.features_pl: features_batch,
                                                                self.labels_pl: labels_batch,
                                                                self.lr_pl: self.learning_rate})
                mean_loss_d += loss_value

            mean_loss_d /= (self.features_train.shape[0] / self.batch_size)

            if last_loss_epoch is not None and mean_loss_d > last_loss_epoch:
                self.learning_rate *= self.decay_factor
                print "learning rate decay: ", self.learning_rate
            else:
                last_loss_epoch = mean_loss_d
            acc, acc_seen, acc_unseen = self.val()
            if acc > best_acc:
                best_acc = acc
                best_acc_seen = acc_seen
                best_acc_unseen = acc_unseen
        return best_acc, best_acc_seen, best_acc_unseen


def get_unique_vector(attributes, labels):
    # get unique class vector
    b = np.ascontiguousarray(labels).view(
        np.dtype((np.void, labels.dtype.itemsize * labels.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return np.flip(attributes[idx], 0), np.flip(labels[idx], 0)


class gmmn(object):
    def __init__(self, args, model_size, data_config):
        self.decay_factor = 0.9
        self.num_epoch = args['num_epoch']
        self.noise_size = data_config['noise_size']
        self.nb_val_loop = args['nb_val_loop']
        self.model_size = model_size
        self.drop_out_rate = args['drop_out_rate']
        self.drop_out_rate_input = args['drop_out_rate_input']
        self.validation = args['validation']
        self.generalized = args['generalized']
        self.best_acc = 0.0
        self.save_var_dict = {}
        self.name = args['data_set']
        self.sigma = data_config['sigma']
        # weight file name (erase when better accuracy)
        self.last_file_name = ""
        self.nb_fake_img = data_config['nb_fake_img']
        self.learning_rate = args['learning_rate']

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

        if self.validation:
            self.features = min_max_scaler.fit_transform(np.load(data_config['dir_path'] + "" + data_config['img_train_file']))
            self.attributes = np.load(data_config['dir_path'] + "" + data_config['attribute_train_file'])
            self.labels = np.load(data_config['dir_path'] + "" + data_config['label_train_file'])
            self.features_test = min_max_scaler.transform(np.load(data_config['dir_path'] + "" + data_config['img_val_file']))
            self.attributes_test = np.load(data_config['dir_path'] + "" + data_config['attribute_val_file'])
            self.labels_test = np.load(data_config['dir_path'] + "" + data_config['label_val_file'])
        else:
            self.features = min_max_scaler.fit_transform(np.load(data_config['dir_path'] + "" + data_config['img_trainval_file']))
            self.attributes = np.load(data_config['dir_path'] + "" + data_config['attribute_trainval_file'])
            self.labels = np.load(data_config['dir_path'] + "" + data_config['label_trainval_file'])

            features_test_unseen = min_max_scaler.transform(np.load(data_config['dir_path'] + "" + data_config['img_test_unseen_file']))
            attributes_test_unseen = np.load(data_config['dir_path'] + "" + data_config['attribute_test_unseen_file'])
            labels_test_unseen = np.load(data_config['dir_path'] + "" + data_config['label_test_unseen_file'])
            self.unique_attributes_test_unseen, self.unique_labels_test_unseen = get_unique_vector(attributes_test_unseen, labels_test_unseen)
            if self.generalized:
                features_test_seen = min_max_scaler.transform(np.load(data_config['dir_path'] + "" + data_config['img_test_seen_file']))
                attributes_test_seen = np.load(data_config['dir_path'] + "" + data_config['attribute_test_seen_file'])
                labels_test_seen = np.load(data_config['dir_path'] + "" + data_config['label_test_seen_file'])
                self.unique_attributes_test_seen, self.unique_labels_test_seen = get_unique_vector(attributes_test_seen, labels_test_seen)
                self.features_test = np.concatenate([features_test_seen, features_test_unseen], axis=0)
                self.attributes_test = np.concatenate([attributes_test_seen, attributes_test_unseen], axis=0)
                self.labels_test = np.concatenate([labels_test_seen, labels_test_unseen], axis=0)
            else:
                self.features_test = features_test_unseen
                self.attributes_test = attributes_test_unseen
                self.labels_test = labels_test_unseen


        print np.mean(np.sum(self.labels, axis=0)[np.sum(self.labels, axis=0) != 0])
        # for batch selection
        self.unique_attributes_train, self.unique_labels_train = get_unique_vector(self.attributes, self.labels)
        # for fake data generation
        self.unique_attributes_test, self.unique_labels_test = get_unique_vector(self.attributes_test, self.labels_test)

        self.batch_size = int(np.amin(np.sum(self.labels, axis=0)[np.sum(self.labels, axis=0) != 0]))

        # img features pl
        self.x_pl = tf.placeholder(tf.float32, shape=(None, self.features.shape[1]))
        # noise pl
        self.z_pl = tf.placeholder(tf.float32, shape=(None, self.noise_size))
        self.attributes_pl = tf.placeholder(tf.float32, shape=(None, self.attributes.shape[1]))
        self.labels_pl = tf.placeholder(tf.int32, shape=(None))
        self.batch_size_pl = tf.placeholder(tf.int32)
        self.drop_out_rate_pl = tf.placeholder(tf.float32)
        self.drop_out_rate_input_pl = tf.placeholder(tf.float32)
        self.lr_pl = tf.placeholder(tf.float32, shape=(None))

        self._create_model()
        if self.validation:
            self.lin_zsl = linear_classifier_zsl(self.validation, self.generalized, self.features_test, self.labels_test, self.unique_labels_test)
        else:
            if self.generalized:
                self.lin_zsl = linear_classifier_zsl(self.validation, self.generalized, self.features_test, self.labels_test, self.unique_labels_test, self.unique_labels_test_unseen, self.unique_labels_test_seen)
            else:
                self.lin_zsl = linear_classifier_zsl(self.validation, self.generalized, self.features_test, self.labels_test, self.unique_labels_test, self.unique_labels_test_unseen)

    def _create_model(self):
        with tf.variable_scope('D'):
            self.decode = self.decoder(self.attributes_pl, self.z_pl)

        X = tf.concat([self.decode, self.x_pl], 0)
        XX = tf.matmul(X, tf.transpose(X))
        X2 = tf.reduce_sum(X * X, 1, keep_dims=True)
        exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)
        s = self.makeScaleMatrix(self.batch_size, self.batch_size)
        S = tf.matmul(s, tf.transpose(s))
        self.loss_d = 0
        for i in range(len(self.sigma)):
            kernel_val = tf.exp(1.0 / self.sigma[i] * exponent)
            self.loss_d += tf.reduce_sum(S * kernel_val)
        self.loss_d = tf.sqrt(self.loss_d)
        self.opt_d = self.optimizer(self.loss_d, self.lr_pl)

    def makeScaleMatrix(self, num_gen, num_orig):
        # first 'N' entries have '1/N', next 'M' entries have '-1/M'
        s1 = tf.constant(1.0 / num_gen, shape=[num_gen, 1])
        s2 = -tf.constant(1.0 / num_orig, shape=[num_orig, 1])
        return tf.concat([s1, s2], 0)

    def decoder(self, attributes, code):
        input = tf.concat([attributes, code], 1)
        for i, size in enumerate(self.model_size):
            input_lin, w, b = self.linear(input, size, name='d'+str(i))
            input = tf.nn.dropout(self.lrelu(input_lin), 1.0-self.drop_out_rate_pl)
        h, w, b = self.linear(input, self.features.shape[1], name='d'+str(len(self.model_size)))
        return h

    def lrelu(self, x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def linear(self, input, output_dim, name=None, stddev=0.01):
        print name
        with tf.variable_scope(name or 'linear'):
            norm = tf.random_normal_initializer(stddev=stddev)
            const = tf.constant_initializer(0.0)
            w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
            b = tf.get_variable('b', [output_dim], initializer=const)
            self.save_var_dict[(name, 0)] = w
            self.save_var_dict[(name, 1)] = b
            return tf.matmul(input, w) + b, w, b

    def optimizer(self, loss, lr):
        batch = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=batch)
        return optimizer

    def next_batch_by_class(self, id_class):
        if id_class == 0:
            idx = np.r_[:self.unique_labels_train.shape[0]]
            random.shuffle(idx)
            self.unique_labels_train = self.unique_labels_train[idx]
            self.unique_attributes_train = self.unique_attributes_train[idx]
        arg = np.where(np.all(self.labels == self.unique_labels_train[id_class], axis=1))[0]
        if arg.shape[0] > self.batch_size:
            arg = np.random.choice(arg, size=self.batch_size, replace=False)
        return self.features[arg], self.attributes[arg], self.labels[arg]

    def train(self):
        with tf.Session() as self.session:
            tf.global_variables_initializer().run()
            last_loss_epoch = None
            for epoch in xrange(self.num_epoch):
                mean_loss_d = 0.0
                for count in xrange(0, self.unique_labels_train.shape[0]):
                     features_batch, attributes_batch, labels_batch = self.next_batch_by_class(count)
                     noise_batch = np.random.normal(0, 1, (features_batch.shape[0], self.noise_size))
                     # update discriminator
                     loss_d, _ = self.session.run([self.loss_d, self.opt_d], {
                                     self.x_pl: features_batch,
                                     self.attributes_pl: attributes_batch,
                                     self.z_pl: noise_batch,
                                     self.batch_size_pl: features_batch.shape[0],
                                     self.drop_out_rate_input_pl: self.drop_out_rate_input,
                                     self.drop_out_rate_pl: self.drop_out_rate,
                                     self.lr_pl: self.learning_rate})
                     mean_loss_d += loss_d

                mean_loss_d /= (self.features.shape[0] / self.batch_size)

                print 'epoch : {}: D : {}'.format(epoch, mean_loss_d)
                if epoch >=20:
                    self.val()

                if last_loss_epoch is not None and mean_loss_d > last_loss_epoch:
                    self.learning_rate *= self.decay_factor
                    print "learning rate decay: ", self.learning_rate
                else:
                    last_loss_epoch = mean_loss_d

                print "-----"

            self.session.close()

    def generate_fake_image_val(self, attributes_class, labels_class, nb_ex):
        features = np.zeros((nb_ex * labels_class.shape[0], self.features.shape[1]))
        labels = np.zeros((nb_ex * labels_class.shape[0], labels_class.shape[1]))
        attributes = np.zeros((nb_ex * attributes_class.shape[0], attributes_class.shape[1]))
        for c in xrange(labels_class.shape[0]):
            noise = np.random.normal(0, 1, (nb_ex, self.noise_size))
            features[c * nb_ex:(c * nb_ex) + nb_ex] = self.session.run(self.decode, {
                self.z_pl:  noise,
                self.attributes_pl: np.tile(attributes_class[c], (nb_ex, 1)),
                self.drop_out_rate_input_pl: 0.0,
                self.drop_out_rate_pl: 0.0})
            labels[c * nb_ex:(c * nb_ex) + nb_ex] = np.tile(labels_class[c], (nb_ex, 1))
            attributes[c * nb_ex:(c * nb_ex) + nb_ex] = np.tile(attributes_class[c], (nb_ex, 1))
        return features, attributes, labels

    def val(self):
        acc = acc_seen = acc_unseen = 0.0
        for l in xrange(self.nb_val_loop):

            features, attributes_temp, labels = self.generate_fake_image_val(self.unique_attributes_test_unseen, self.unique_labels_test_unseen, self.nb_fake_img)
            if self.generalized:
                features = np.concatenate([features, self.features], axis=0)
                labels = np.concatenate([labels, self.labels], axis=0)

            with tf.Session() as linear_sess_zsl:
                best_acc, best_acc_seen, best_acc_unseen = self.lin_zsl.learn(linear_sess_zsl, features, labels)
                acc += best_acc
                acc_seen += best_acc_seen
                acc_unseen += best_acc_unseen
        acc = acc / self.nb_val_loop
        acc_seen = acc_seen / self.nb_val_loop
        acc_unseen = acc_unseen / self.nb_val_loop
        if self.validation:
            print 'validation accuracy : unseen class : ', acc
        else:
            if self.generalized:
                print 'accuracy : seen class : ', acc_seen, '| unseen class : ', acc_unseen, '| harmonic : ', acc
            else:
                print 'accuracy : unseen class : ', acc

        if acc > self.best_acc:
            if self.best_acc != 0.0:
                os.remove(self.last_file_name + ".npy")
            self.best_acc = acc
            self.last_file_name = "model_weights_gmmn/" + self.name + "_" \
                                  + str(np.around(self.learning_rate, decimals=2)) + "_" \
                                  + str(self.model_size) + "_" \
                                  + str(self.noise_size) + "_" \
                                  + str(self.sigma) + "_"  \
                                  + str(self.drop_out_rate_input) + "_" \
                                  + str(self.drop_out_rate) + "_" \
                                  + str(self.nb_fake_img) + "_" \
                                  + str(np.around(acc_unseen, decimals=4)) + "_" \
                                  + str(np.around(acc_seen, decimals=4)) + "_" \
                                  + str(np.around(self.best_acc, decimals=4))
            self.save_npy(self.session, self.last_file_name)


    def save_npy(self, sess, npy_path):
        assert isinstance(sess, tf.Session)
        data_dict = {}
        for (name, idx), var in self.save_var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out
        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path



def main():
    model_size = [8192, 8192]
    sigma = [2, 5, 10, 20, 40, 80]
    parser = argparse.ArgumentParser(description="Run GMMN experiments on various ZSL datasets")
    parser.add_argument('--data_set', type=str, default="AWA1",
                        help="Dataset to use. Available:\n"
                             "AWA1: Animals with Attributes 1, "
                             "AWA2: Animals with Attributes 2, "
                             "CUB: Caltech-UCSD Birds-200-2011, "
                             "APY: aYahoo and aPascal, "
                             "SUN: SUN Attribute Database")
    parser.add_argument('--num_epoch', type=int, default=2000, help="Number of epoch")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="Starting learning rate")
    parser.add_argument('--noise_size', type=int, default=15, help="Input noise size")
    parser.add_argument('--drop_out_rate_input', type=float, default=0.2, help="Drop out input on layer")
    parser.add_argument('--drop_out_rate', type=float, default=0.5, help="Drop out on hidden layer")
    parser.add_argument('--validation', type=bool, default=False, help="Validation stage, use validation split")
    parser.add_argument('--generalized', type=bool, default=False, help="Generalized zero shot learning")
    parser.add_argument('--nb_img', type=int, default=50, help="Number of generated image feature by unseen class")

    pars_args = parser.parse_args()

    args = {'data_set': pars_args.data_set,
            'num_epoch': pars_args.num_epoch,
            'learning_rate': pars_args.learning_rate,
            'validation': pars_args.validation,
            'generalized': pars_args.generalized,
            # number of accuracy test per validation
            'nb_val_loop': 1,
            'drop_out_rate': pars_args.drop_out_rate,
            'drop_out_rate_input': pars_args.drop_out_rate_input}

    DIR_PATH = "data/"
    if args['data_set'] == "AWA1":
        data_config = {'dir_path': DIR_PATH + "AWA1/",
                     'img_trainval_file': "AWA1_trainval_features.npy",
                     'attribute_trainval_file': "AWA1_trainval_attributes.npy",
                     'label_trainval_file': "AWA1_trainval_labels.npy",
                     'img_test_seen_file': "AWA1_test_seen_features.npy",
                     'attribute_test_seen_file': "AWA1_test_seen_attributes.npy",
                     'label_test_seen_file': "AWA1_test_seen_labels.npy",
                     'img_test_unseen_file': "AWA1_test_unseen_features.npy",
                     'attribute_test_unseen_file': "AWA1_test_unseen_attributes.npy",
                     'label_test_unseen_file': "AWA1_test_unseen_labels.npy",
                     'img_train_file': "AWA1_train_features.npy",
                     'attribute_train_file': "AWA1_train_attributes.npy",
                     'label_train_file': "AWA1_train_labels.npy",
                     'img_val_file': "AWA1_val_features.npy",
                     'attribute_val_file': "AWA1_val_attributes.npy",
                     'label_val_file': "AWA1_val_labels.npy",
                     'noise_size': pars_args.noise_size,
                     'sigma': sigma,
                     'nb_fake_img': pars_args.nb_img}
    elif args['data_set'] == "AWA2":
        data_config = {'dir_path': DIR_PATH + "AWA2/",
                     'img_trainval_file': "AWA2_trainval_features.npy",
                     'attribute_trainval_file': "AWA2_trainval_attributes.npy",
                     'label_trainval_file': "AWA2_trainval_labels.npy",
                     'img_test_seen_file': "AWA2_test_seen_features.npy",
                     'attribute_test_seen_file': "AWA2_test_seen_attributes.npy",
                     'label_test_seen_file': "AWA2_test_seen_labels.npy",
                     'img_test_unseen_file': "AWA2_test_unseen_features.npy",
                     'attribute_test_unseen_file': "AWA2_test_unseen_attributes.npy",
                     'label_test_unseen_file': "AWA2_test_unseen_labels.npy",
                     'img_train_file': "AWA2_train_features.npy",
                     'attribute_train_file': "AWA2_train_attributes.npy",
                     'label_train_file': "AWA2_train_labels.npy",
                     'img_val_file': "AWA2_val_features.npy",
                     'attribute_val_file': "AWA2_val_attributes.npy",
                     'label_val_file': "AWA2_val_labels.npy",
                     'noise_size': pars_args.noise_size,
                     'sigma': sigma,
                     'nb_fake_img': pars_args.nb_img}
    elif args['data_set'] == "APY":
        data_config = {'dir_path': DIR_PATH + "APY/",
                     'img_trainval_file': "APY_trainval_features.npy",
                     'attribute_trainval_file': "APY_trainval_attributes.npy",
                     'label_trainval_file': "APY_trainval_labels.npy",
                     'img_test_seen_file': "APY_test_seen_features.npy",
                     'attribute_test_seen_file': "APY_test_seen_attributes.npy",
                     'label_test_seen_file': "APY_test_seen_labels.npy",
                     'img_test_unseen_file': "APY_test_unseen_features.npy",
                     'attribute_test_unseen_file': "APY_test_unseen_attributes.npy",
                     'label_test_unseen_file': "APY_test_unseen_labels.npy",
                     'img_train_file': "APY_train_features.npy",
                     'attribute_train_file': "APY_train_attributes.npy",
                     'label_train_file': "APY_train_labels.npy",
                     'img_val_file': "APY_val_features.npy",
                     'attribute_val_file': "APY_val_attributes.npy",
                     'label_val_file': "APY_val_labels.npy",
                     'noise_size': pars_args.noise_size,
                     'sigma': sigma,
                     'nb_fake_img': pars_args.nb_img}
    elif args['data_set'] == "SUN":
        data_config = {'dir_path': DIR_PATH + "SUN/",
                     'img_trainval_file': "SUN_trainval_features.npy",
                     'attribute_trainval_file': "SUN_trainval_attributes.npy",
                     'label_trainval_file': "SUN_trainval_labels.npy",
                     'img_test_seen_file': "SUN_test_seen_features.npy",
                     'attribute_test_seen_file': "SUN_test_seen_attributes.npy",
                     'label_test_seen_file': "SUN_test_seen_labels.npy",
                     'img_test_unseen_file': "SUN_test_unseen_features.npy",
                     'attribute_test_unseen_file': "SUN_test_unseen_attributes.npy",
                     'label_test_unseen_file': "SUN_test_unseen_labels.npy",
                     'img_train_file': "SUN_train_features.npy",
                     'attribute_train_file': "SUN_train_attributes.npy",
                     'label_train_file': "SUN_train_labels.npy",
                     'img_val_file': "SUN_val_features.npy",
                     'attribute_val_file': "SUN_val_attributes.npy",
                     'label_val_file': "SUN_val_labels.npy",
                     'noise_size': pars_args.noise_size,
                     'sigma': sigma,
                     'nb_fake_img': pars_args.nb_img}
    elif args['data_set'] == "CUB":
        data_config = {'dir_path': DIR_PATH + "CUB/",
                     'img_trainval_file': "CUB_trainval_features.npy",
                     'attribute_trainval_file': "CUB_trainval_attributes.npy",
                     'label_trainval_file': "CUB_trainval_labels.npy",
                     'img_test_seen_file': "CUB_test_seen_features.npy",
                     'attribute_test_seen_file': "CUB_test_seen_attributes.npy",
                     'label_test_seen_file': "CUB_test_seen_labels.npy",
                     'img_test_unseen_file': "CUB_test_unseen_features.npy",
                     'attribute_test_unseen_file': "CUB_test_unseen_attributes.npy",
                     'label_test_unseen_file': "CUB_test_unseen_labels.npy",
                     'img_train_file': "CUB_train_features.npy",
                     'attribute_train_file': "CUB_train_attributes.npy",
                     'label_train_file': "CUB_train_labels.npy",
                     'img_val_file': "CUB_val_features.npy",
                     'attribute_val_file': "CUB_val_attributes.npy",
                     'label_val_file': "CUB_val_labels.npy",
                     'noise_size': pars_args.noise_size,
                     'sigma': sigma,
                     'nb_fake_img':   pars_args.nb_img}

    model = gmmn(args, model_size, data_config)
    model.train()
    # reset graph and variables
    tf.reset_default_graph()

if __name__ == '__main__':
    main()
