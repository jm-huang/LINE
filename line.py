import tensorflow as tf
import numpy as np
import os
import argparse
import logging
from utils import evaluate, DataLoader, get_labels
import time
FORMAT = "%(asctime)-15s %(name)-10s %(levelname)-5s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    # inputs
    arg_parser.add_argument("--link", required=True, type=str,
            help="filecontaining the links(assume as undirected)")
    arg_parser.add_argument("--label", type=str, help="filecontaining the node labels")
    arg_parser.add_argument("--node_count", type=int,
                            help="the total number of nodes", required=True)
    # output
    arg_parser.add_argument("--save", type=str, default="",
            help="Directory to write the model, training summaries and node embeddings.")
    # hyper params:
    arg_parser.add_argument("--embedding_size", type=int,
                            help="embedding dimention", required=True)
    arg_parser.add_argument("--learning_rate", type=float, required=True)
    arg_parser.add_argument("--num_batches", type=int, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=1024, help="batch_size")
    arg_parser.add_argument("--negative", type=int, required=True, help="negative ratio")

    args = arg_parser.parse_args()
    if args.save != "" and args.save is not None and not os.path.exists(args.save):
        os.makedirs(args.save)
    return args


class LINE(object):
    def __init__(self, options, session, node_count, total_samples):
        self._options = options
        self._session = session
        logging.info("total_samples {}".format(total_samples))
        # self._neg_sampler = NegSampling(network, NEG_SAMPLING_POWER, 1314)
        self.total_samples = total_samples
        self.build_graph(node_count)

    def build_graph(self, node_count):
        """
        build the LINE model
        """
        init_width = 0.5 / self._options.embedding_size

        d = tf.random_uniform([node_count, self._options.embedding_size],
                              -init_width, init_width)
        self._embs = tf.Variable(d, name="embs")
        self._cur_samples = tf.Variable(0, name="cur_samples")

        # d = tf.random_uniform([self._options.embedding_size, self._options.embedding_size],
        #                       -init_width, init_width)
        # self._attention_matrix = tf.Variable(d, name="attention_matrix")

        self._global_step = tf.Variable(0, name="global_step")

        self._ph_source = tf.placeholder(dtype=tf.int32, name="source")
        self._ph_target = tf.placeholder(dtype=tf.int32, name="target")
        self._ph_label  = tf.placeholder(dtype=tf.float32, name="label")
        self._ph_neighbors = tf.placeholder(dtype=tf.int32, shape=None, name="target_neighbors")

        update_cur_samples = self._cur_samples.assign(
                tf.size(self._ph_source) + self._cur_samples)
        self._update_cur_samples = update_cur_samples
        # reset_cur_samples = self._cur_samples.assign(0)
        # self._reset_cur_samples = reset_cur_samples

        source_emb = tf.nn.embedding_lookup(self._embs, self._ph_source)
        target_emb = tf.nn.embedding_lookup(self._embs, self._ph_target)
        loss = self.make_loss(source_emb, target_emb)
        self._loss = loss
        self._train = self.make_train_op()

        init = tf.global_variables_initializer()
        self._session.run(init)
        init = tf.local_variables_initializer()
        self._session.run(init)

        self._saver = tf.train.Saver()

    def make_loss(self, source_emb, target_emb):
        sigma = tf.reduce_sum(tf.multiply(source_emb, target_emb), axis=1)
        sigma = tf.multiply(sigma, self._ph_label)
        # tf.nn.softplus(-x) = -tf.log_sigmoid(x) = -log(1 / (1 + exp(-x)))
        loss = tf.reduce_mean(tf.nn.softplus(-sigma))
        return loss

    def make_train_op(self):
        lr = self._options.learning_rate * tf.maximum(
             0.0001, 1.0 - tf.cast(self._cur_samples, tf.float32) / self.total_samples)
        self._lr = lr
        # opt = tf.train.GradientDescentOptimizer(lr)
        # opt = tf.train.AdamOptimizer(learning_rate=self._options.learning_rate)
        opt = tf.train.RMSPropOptimizer(learning_rate=lr)
        train_op = opt.minimize(self._loss,
                             global_step=self._global_step,
                             gate_gradients=opt.GATE_NONE)
        return train_op

    def train(self, data_loader, node_labels=None):
        """train LINE model
        Args:
            data_loader: train samples generater.
            labels(ndarray, shape=(node_count, label)):
                the node labels, `nid lid` for each column, respectively.
        """
        opts = self._options
        sampling_time = 0
        train_time = 0
        logging.info("batch\tloss\tsampling_time\ttrain_time\ttrain_acc\ttest_acc")
        for b in xrange(opts.num_batches):
            t0 = time.time()
            source, target, label = data_loader.fetch_batch2(opts.batch_size,
                                                             opts.negative,
                                                             edge_sampling="uniform",
                                                             node_sampling="uniform")
            t1 = time.time()
            sampling_time += t1 - t0

            feed_dict = {self._ph_source: source,
                         self._ph_target: target,
                         self._ph_label:  label}

            t0 = time.time()
            self._session.run(
                    [self._train, self._update_cur_samples],
                    feed_dict=feed_dict)
            t1 = time.time()
            train_time += t1 - t0

            if b % 10000 == 0 or b == opts.num_batches - 1:
                loss = self._session.run(self._loss, feed_dict=feed_dict)
                if node_labels is not None:
                    embedding = self.get_embs()
                    emb_map = data_loader.embedding_mapping(embedding)
                    sorted_emb = sorted(emb_map.items(), cmp=lambda a, b: int.__cmp__(a[0], b[0]))
                    embs = [e[1] for e in sorted_emb]
                    # pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                    #             open('data/embedding_%s.pkl' % suffix, 'wb'))
                    train_acc, test_acc = evaluate(np.array(embs), node_labels)
                else:
                    train_acc, test_acc = 0.0, 0.0
                logging.info('{}M\t{:.4f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                             b / 10000, loss, sampling_time, train_time, train_acc, test_acc))
                sampling_time, train_time = 0, 0

    def get_embs(self):
        """
        Return:
            node_embs(ndarray, shape=(node_size, dim_size))
        """
        embs = self._session.run(self._embs)
        norm_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        return norm_embs


def main():
    args = parse_args()
    logging.info("\n\targs: {}\n".format(args))
    labels = None
    if args.label is not None:
        logging.info("Init labels ...")
        labels = get_labels(args.label)
    logging.info("Init data loader ...")
    data_loader = DataLoader(args.link, args.node_count)
    with tf.Graph().as_default(), tf.Session() as session:
        # logging.info("loading train data")
        # train_samps = np.loadtxt(args.train_data, dtype=int)
        # train_samps = get_sample(args.train_data)

        total_samples = args.batch_size * args.num_batches
        logging.info("Initing LINE model")
        model = LINE(args, session, args.node_count, total_samples)
        logging.info("training ...")
        model.train(data_loader, labels)

        if args.save != "":
            logging.info("Saving ...")
            model._saver.save(session,
                              os.path.join(args.save, "model.ckpt"),
                              global_step=args.epochs)


if __name__ == "__main__":
    main()
