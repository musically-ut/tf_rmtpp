import tensorflow as tf
import numpy as np
import os
import decorated_options as Deco
from .utils import create_dir, variable_summaries


def_opts = Deco.Options(
    hidden_layer_size=64,   # 64, 128, 256, 512, 1024
    batch_size=64,          # 16, 32, 64
    learning_rate=0.1,      # 0.1, 0.01, 0.001
    momentum=0.9,
    l2_penalty=0.001,
    embed_size=64,
    float_type=tf.float32,
    seed=42,
    scope='RMTPP',
    save_dir='./save.rmtpp/',
    summary_dir='./summary.rmtpp/',

    device_gpu='/gpu:0',
    device_cpu='/cpu:0',

    bptt=10
)


class RMTPP:
    """Class implementing the Recurrent Marked Temporal Point Process model."""

    @Deco.optioned()
    def __init__(self, sess, num_categories, hidden_layer_size, batch_size,
                 learning_rate, momentum, l2_penalty, embed_size,
                 float_type, bptt, seed, scope, save_dir,
                 device_gpu, device_cpu, summary_dir):
        self.HIDDEN_LAYER_SIZE = hidden_layer_size
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.MOMENTUM = momentum
        self.L2_PENALTY = l2_penalty
        self.EMBED_SIZE = embed_size
        self.BPTT = bptt
        self.SAVE_DIR = save_dir
        self.SUMMARY_DIR = summary_dir

        self.NUM_CATEGORIES = num_categories
        self.FLOAT_TYPE = float_type

        self.DEVICE_CPU = device_cpu
        self.DEVICE_GPU = device_gpu

        self.sess = sess
        self.seed = seed
        self.last_epoch = 0

        with tf.variable_scope(scope):
            with tf.device(device_gpu):
                # Make input variables
                self.events_in = tf.placeholder(tf.int32, [None, self.BPTT], name='events_in')
                self.times_in = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_in')

                self.events_out = tf.placeholder(tf.int32, [None, self.BPTT], name='events_out')
                self.times_out = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_out')

                self.inf_batch_size = tf.shape(self.events_in)[0]

                # Make variables
                with tf.variable_scope('hidden_state'):
                    self.Wt = tf.get_variable(name='Wt',
                                              shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(0.0))
                    # The first row of Wem is merely a placeholder (will not be trained).
                    self.Wem = tf.get_variable(name='Wem', shape=(self.NUM_CATEGORIES + 1, self.EMBED_SIZE),
                                               dtype=self.FLOAT_TYPE,
                                               initializer=tf.constant_initializer(0.0))
                    self.Wh = tf.get_variable(name='Wh', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(np.eye(self.HIDDEN_LAYER_SIZE)))
                    self.bh = tf.get_variable(name='bh', shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(0.0))

                with tf.variable_scope('output'):
                    self.wt = tf.get_variable(name='wt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(1.0))

                    self.Wy = tf.get_variable(name='Wy', shape=(self.EMBED_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(0.0))

                    # The first column of Vy is merely a placeholder (will not be trained).
                    self.Vy = tf.get_variable(name='Vy', shape=(self.HIDDEN_LAYER_SIZE, self.NUM_CATEGORIES + 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(0.0))
                    self.Vt = tf.get_variable(name='Vt', shape=(self.HIDDEN_LAYER_SIZE, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(0.0))
                    self.bt = tf.get_variable(name='bt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(0.0))
                    self.bk = tf.get_variable(name='bk', shape=(1, self.NUM_CATEGORIES + 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(0.0))

                self.all_vars = [self.Wt, self.Wem, self.Wh, self.bh,
                                 self.wt, self.Wy, self.Vy, self.Vt, self.bt, self.bk]

                # Add summaries for all (trainable) variables
                for v in self.all_vars:
                    variable_summaries(v)

                # Make graph
                # RNNcell = RNN_CELL_TYPE(HIDDEN_LAYER_SIZE)

                # Initial state for GRU cells
                self.initial_state = state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE], dtype=self.FLOAT_TYPE, name='hidden_state')

                self.loss = 0.0
                batch_ones = tf.ones((self.inf_batch_size, 1), dtype=self.FLOAT_TYPE)

                self.hidden_states = []
                self.new_hidden_states = []
                self.event_preds = []

                self.time_losses = []
                self.mark_losses = []
                self.delta_ts = []

                with tf.name_scope('BPTT'):
                    for i in range(self.BPTT):
                        events_embedded = tf.nn.embedding_lookup(self.Wem, self.events_in[:, i])
                        delta_t = tf.expand_dims(self.times_in[:, i], axis=-1)

                        # output, state = RNNcell(events_embedded, state)

                        # TODO Does TF automatically broadcast? Then we'll not need multiplication with tf.ones
                        with tf.name_scope('state_recursion'):
                            new_state = tf.clip_by_value(
                                                 tf.matmul(state, self.Wh) +
                                                 tf.matmul(events_embedded, self.Wy) +
                                                 tf.matmul(delta_t, self.Wt) +
                                                 tf.matmul(batch_ones, self.bh),
                                                 0.0, 1e6, name='h_t')
                            state = tf.where(self.events_in[:, i] > 0, new_state, state)

                        with tf.name_scope('loss_calc'):
                            base_intensity = tf.matmul(batch_ones, self.bt)
                            log_lambda_ = (tf.matmul(state, self.Vt) +
                                           delta_t * self.wt +
                                           base_intensity)

                            lambda_ = tf.exp(tf.minimum(50.0, log_lambda_), name='lambda_')
                            wt_non_zero = tf.sign(self.wt) * tf.maximum(1e-6, tf.abs(self.wt))
                            log_f_star = (log_lambda_ +
                                          (1.0 / wt_non_zero) * tf.exp(tf.minimum(50.0, tf.matmul(state, self.Vt) + base_intensity)) -
                                          (1.0 / wt_non_zero) * lambda_)

                            events_pred = tf.nn.softmax(
                                tf.minimum(50.0,
                                           tf.matmul(state, self.Vy) + batch_ones * self.bk),
                                name='Pr_events'
                            )

                            time_loss = log_f_star
                            mark_loss = tf.expand_dims(
                                tf.log(
                                    tf.maximum(
                                        1e-6,
                                        tf.gather_nd(
                                            events_pred,
                                            tf.concat([
                                                tf.expand_dims(tf.range(self.inf_batch_size), -1),
                                                tf.expand_dims(self.events_out[:, i], -1)
                                            ], axis=1, name='Pr_next_event'
                                            )
                                        )
                                    )
                                ), axis=-1, name='log_Pr_next_event'
                            )
                            step_loss = time_loss + mark_loss

                            # In the batch some of the sequences may have ended before we get to the
                            # end of the seq. In such cases, the events will be zero.
                            # TODO Figure out how to do this with RNNCell, LSTM, etc.
                            num_events = tf.reduce_sum(tf.where(self.events_in[:, i] > 0,
                                                       tf.ones(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE),
                                                       tf.zeros(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE)),
                                                       name='num_events')

                            self.loss -= tf.cond(num_events > 0,
                                                 lambda: tf.reduce_sum(
                                                     tf.where(self.events_in[:, i] > 0,
                                                              tf.squeeze(step_loss) / num_events,
                                                              tf.zeros(shape=(self.inf_batch_size,))),
                                                     name='batch_bptt_loss'),
                                                 lambda: 0.0)

                        self.time_losses.append(time_loss)
                        self.mark_losses.append(mark_loss)

                        self.hidden_states.append(state)
                        self.new_hidden_states.append(new_state)
                        self.event_preds.append(events_pred)

                        self.delta_ts.append(delta_t)

                self.final_state = self.hidden_states[-1]

                with tf.device(device_cpu):
                    # Global step needs to be on the CPU (Why?)
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.learning_rate = tf.train.inverse_time_decay(self.LEARNING_RATE,
                                                                 global_step=self.global_step,
                                                                 decay_steps=10.0,
                                                                 decay_rate=.001)
                # self.global_step is incremented automatically by the
                # optimizer.

                # self.increment_global_step = tf.assign(
                #     self.global_step,
                #     self.global_step + 1,
                #     name='update_global_step'
                # )

                # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                          beta1=self.MOMENTUM)

                # Capping the gradient before minimizing.
                # update = optimizer.minimize(loss)

                # Performing manual gradient clipping.
                self.gvs = self.optimizer.compute_gradients(self.loss)
                # update = optimizer.apply_gradients(gvs)

                # capped_gvs = [(tf.clip_by_norm(grad, 100.0), var) for grad, var in gvs]
                grads, vars_ = list(zip(*self.gvs))

                for g, v in zip(grads, vars_):
                    variable_summaries(g, name='grad-' + v.name.split('/')[-1][:-2])

                variable_summaries(self.hidden_states, name='agg-hidden-states')
                variable_summaries(self.event_preds, name='agg-event-preds-softmax')
                variable_summaries(self.time_losses, name='agg-time-losses')
                variable_summaries(self.mark_losses, name='agg-mark-losses')
                variable_summaries(self.mark_losses, name='agg-delta-ts')
                variable_summaries(self.new_hidden_states, name='agg-new-hidden-states')

                self.norm_grads, self.global_norm = tf.clip_by_global_norm(grads, 100.0)
                capped_gvs = list(zip(self.norm_grads, vars_))

                self.update = self.optimizer.apply_gradients(capped_gvs,
                                                             global_step=self.global_step)

                self.tf_init = tf.global_variables_initializer()
                self.tf_merged_summaries = tf.summary.merge_all()
                # self.check_nan = tf.add_check_numerics_ops()

    def initialize(self, finalize=False):
        """Initialize the global trainable variables."""
        self.sess.run(self.tf_init)

        if finalize:
            # This prevents memory leaks by disallowing changes to the graph
            # after initialization.
            self.sess.graph.finalize()

    def train(self, training_data, num_epochs=1,
              restart=False, check_nans=False, one_batch=False,
              with_summaries=False):
        """Train the model given the training data."""
        create_dir(self.SAVE_DIR)
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)

        # TODO: Should give the variable list explicitly for RMTPP only, in case
        # There are variables outside RMTPP model.
        # TODO: Why does this create new nodes in the graph? Possibly memory leak?
        saver = tf.train.Saver(tf.global_variables())

        train_writer = tf.summary.FileWriter(self.SUMMARY_DIR + '/train',
                                             self.sess.graph)

        rs = np.random.RandomState(seed=self.seed)

        if ckpt and restart:
            print('Restoring from {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        train_event_in_seq = training_data['train_event_in_seq']
        train_time_in_seq = training_data['train_time_in_seq']
        train_event_out_seq = training_data['train_event_out_seq']
        train_time_out_seq = training_data['train_event_out_seq']

        idxes = list(range(len(train_event_in_seq)))
        n_batches = len(idxes) // self.BATCH_SIZE

        for epoch in range(self.last_epoch, self.last_epoch + num_epochs):
            rs.shuffle(idxes)

            print("Starting epoch...", epoch)
            total_loss = 0.0

            for batch_idx in range(n_batches):
                # TODO: This is horribly inefficient. Move this to a separate thread using FIFOQueues.
                batch_idxes = idxes[batch_idx * self.BATCH_SIZE:(batch_idx + 1) * self.BATCH_SIZE]
                batch_event_train_in = train_event_in_seq[batch_idxes, :]
                batch_event_train_out = train_event_out_seq[batch_idxes, :]
                batch_time_train_in = train_time_in_seq[batch_idxes, :]
                batch_time_train_out = train_time_out_seq[batch_idxes, :]

                cur_state = np.zeros((self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE))
                batch_loss = 0.0

                for bptt_idx in range(0, len(batch_event_train_in[0]) - self.BPTT, self.BPTT):
                    bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
                    bptt_event_in = batch_event_train_in[:, bptt_range]
                    bptt_event_out = batch_event_train_out[:, bptt_range]
                    bptt_time_in = batch_time_train_in[:, bptt_range]
                    bptt_time_out = batch_time_train_out[:, bptt_range]

                    feed_dict = {
                        self.initial_state: cur_state,
                        self.events_in: bptt_event_in,
                        self.events_out: bptt_event_out,
                        self.times_in: bptt_time_in,
                        self.times_out: bptt_time_out
                    }

                    if check_nans:
                        raise NotImplemented('tf.add_check_numerics_ops is '
                                             'incompatible with tf.cond and '
                                             'tf.while_loop.')
                        # _, _, cur_state, loss_ = \
                        #     self.sess.run([self.check_nan, self.update,
                        #                    self.final_state, self.loss],
                        #                   feed_dict=feed_dict)
                    else:
                        if with_summaries:
                            _, summaries, cur_state, loss_, step = \
                                self.sess.run([self.update,
                                               self.tf_merged_summaries,
                                               self.final_state,
                                               self.loss,
                                               self.global_step],
                                              feed_dict=feed_dict)

                            train_writer.add_summary(summaries, step)
                        else:
                            _, cur_state, loss_ = \
                                self.sess.run([self.update,
                                               self.final_state, self.loss],
                                              feed_dict=feed_dict)
                    batch_loss += loss_

                total_loss += batch_loss
                if batch_idx % 10 == 0:
                    print('Loss during batch {} last BPTT = {:.3f}, lr = {:.5f}'
                          .format(batch_idx, batch_loss, self.sess.run(self.learning_rate)))

            # self.sess.run(self.increment_global_step)
            print('Loss on last epoch = {:.4f}, new lr = {:.5f}, global_step = {}'
                  .format(total_loss / n_batches,
                          self.sess.run(self.learning_rate),
                          self.sess.run(self.global_step)))

            if one_batch:
                print('Breaking after just one batch.')
                break

        checkpoint_path = os.path.join(self.SAVE_DIR, 'model.ckpt')
        saver.save(self.sess, checkpoint_path, global_step=self.global_step)
        print('Model saved at {}'.format(checkpoint_path))

        # Remember how many epochs we have trained.
        self.last_epoch += num_epochs

    def restore(self):
        """Restore the model from saved state."""
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)
        print('Loading the model from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, test_data):
        """Treat the entire test-data as a single batch."""

        test_event_in_seq = test_data['test_event_in_seq']
        test_time_in_seq = test_data['test_time_in_seq']
        # test_time_out_seq = test_data['test_time_out_seq']
        # test_event_out_seq = test_data['test_event_out_seq']

        all_hidden_states = []
        all_event_preds = []

        cur_state = np.zeros((len(test_event_in_seq), self.HIDDEN_LAYER_SIZE))

        for bptt_idx in range(0, len(test_event_in_seq[0]) - self.BPTT, self.BPTT):
            bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
            bptt_event_in = test_event_in_seq[:, bptt_range]
            bptt_time_in = test_time_in_seq[:, bptt_range]

            feed_dict = {
                self.initial_state: cur_state,
                self.events_in: bptt_event_in,
                self.times_in: bptt_time_in
            }

            bptt_hidden_states, bptt_events_pred, cur_state = self.sess.run(
                [self.hidden_states, self.event_preds, self.final_state],
                feed_dict=feed_dict
            )

            all_hidden_states.extend(bptt_hidden_states)
            all_event_preds.extend(bptt_events_pred)

        return all_hidden_states, all_event_preds
