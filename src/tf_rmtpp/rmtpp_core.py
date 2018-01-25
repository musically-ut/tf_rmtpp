import tensorflow as tf
import numpy as np
import os
import decorated_options as Deco
from .utils import create_dir, variable_summaries, MAE, ACC
from scipy.integrate import quad
import multiprocessing as MP


__EMBED_SIZE = 4
__HIDDEN_LAYER_SIZE = 16  # 64, 128, 256, 512, 1024

def_opts = Deco.Options(
    batch_size=64,          # 16, 32, 64

    learning_rate=0.1,      # 0.1, 0.01, 0.001
    momentum=0.9,
    decay_steps=100,
    decay_rate=0.001,

    l2_penalty=0.001,         # Unused

    float_type=tf.float32,

    seed=42,
    scope='RMTPP',
    save_dir='./save.rmtpp/',
    summary_dir='./summary.rmtpp/',

    device_gpu='/gpu:0',
    device_cpu='/cpu:0',

    bptt=20,
    cpu_only=False,

    embed_size=__EMBED_SIZE,
    Wem=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,

    Wt=np.ones((1, __HIDDEN_LAYER_SIZE)) * 1e-3,
    Wh=np.eye(__HIDDEN_LAYER_SIZE),
    bh=np.ones((1, __HIDDEN_LAYER_SIZE)),
    wt=1.0,
    Wy=np.ones((__EMBED_SIZE, __HIDDEN_LAYER_SIZE)) * 0.0,
    Vy=lambda num_categories: np.ones((__HIDDEN_LAYER_SIZE, num_categories)) * 0.001,
    Vt=np.ones((__HIDDEN_LAYER_SIZE, 1)) * 0.001,
    bt=np.log(1.0), # bt is provided by the base_rate
    bk=lambda num_categories: np.ones((1, num_categories)) * 0.0
)


def softplus(x):
    """Numpy counterpart to tf.nn.softplus"""
    return np.log1p(np.exp(x))


def quad_func(t, c, w):
    """This is the t * f(t) function calculating the mean time to next event,
    given c, w."""
    return c * t * np.exp(-w * t + (c / w) * (np.exp(-w * t) - 1))


class RMTPP:
    """Class implementing the Recurrent Marked Temporal Point Process model."""

    @Deco.optioned()
    def __init__(self, sess, num_categories, batch_size,
                 learning_rate, momentum, l2_penalty, embed_size,
                 float_type, bptt, seed, scope, save_dir, decay_steps, decay_rate,
                 device_gpu, device_cpu, summary_dir, cpu_only,
                 Wt, Wem, Wh, bh, wt, Wy, Vy, Vt, bk, bt):
        self.HIDDEN_LAYER_SIZE = Wh.shape[0]
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

        self.rs = np.random.RandomState(seed + 42)

        with tf.variable_scope(scope):
            with tf.device(device_gpu if not cpu_only else device_cpu):
                # Make input variables
                self.events_in = tf.placeholder(tf.int32, [None, self.BPTT], name='events_in')
                self.times_in = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_in')

                self.events_out = tf.placeholder(tf.int32, [None, self.BPTT], name='events_out')
                self.times_out = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_out')

                self.batch_num_events = tf.placeholder(self.FLOAT_TYPE, [], name='bptt_events')

                self.inf_batch_size = tf.shape(self.events_in)[0]

                # Make variables
                with tf.variable_scope('hidden_state'):
                    self.Wt = tf.get_variable(name='Wt',
                                              shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wt))

                    # TODO: Generalize to multiple marks (need to be predicted
                    # for future events) and context for the present event
                    # (which need not be predicted).
                    # self.Wem will be converted to a list of embedding
                    # matrices depending on the number of marks or contexts
                    # each event has.
                    # A similar self.Wctx will also be needed to embed
                    # contextual data.
                    # The marks can then be independently constructed from the
                    # hidden state by a similar list of matrices from self.Wy.
                    self.Wem = tf.get_variable(name='Wem', shape=(self.NUM_CATEGORIES, self.EMBED_SIZE),
                                               dtype=self.FLOAT_TYPE,
                                               initializer=tf.constant_initializer(Wem(self.NUM_CATEGORIES)))
                    self.Wh = tf.get_variable(name='Wh', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wh))
                    self.bh = tf.get_variable(name='bh', shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bh))

                with tf.variable_scope('output'):
                    self.wt = tf.get_variable(name='wt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(wt))

                    self.Wy = tf.get_variable(name='Wy', shape=(self.EMBED_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wy))

                    # The first column of Vy is merely a placeholder (will not be trained).
                    self.Vy = tf.get_variable(name='Vy', shape=(self.HIDDEN_LAYER_SIZE, self.NUM_CATEGORIES),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vy(self.NUM_CATEGORIES)))
                    self.Vt = tf.get_variable(name='Vt', shape=(self.HIDDEN_LAYER_SIZE, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vt))
                    self.bt = tf.get_variable(name='bt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bt))
                    self.bk = tf.get_variable(name='bk', shape=(1, self.NUM_CATEGORIES),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bk(num_categories)))

                self.all_vars = [self.Wt, self.Wem, self.Wh, self.bh,
                                 self.wt, self.Wy, self.Vy, self.Vt, self.bt, self.bk]

                # Add summaries for all (trainable) variables
                with tf.device(device_cpu):
                    for v in self.all_vars:
                        variable_summaries(v)

                # Make graph
                # RNNcell = RNN_CELL_TYPE(HIDDEN_LAYER_SIZE)

                # Initial state for GRU cells
                self.initial_state = state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE],
                                                      dtype=self.FLOAT_TYPE,
                                                      name='initial_state')
                self.initial_time = last_time = tf.zeros((self.inf_batch_size,),
                                                         dtype=self.FLOAT_TYPE,
                                                         name='initial_time')

                self.loss = 0.0
                ones_2d = tf.ones((self.inf_batch_size, 1), dtype=self.FLOAT_TYPE)
                # ones_1d = tf.ones((self.inf_batch_size,), dtype=self.FLOAT_TYPE)

                self.hidden_states = []
                self.event_preds = []

                self.time_LLs = []
                self.mark_LLs = []
                self.log_lambdas = []
                # self.delta_ts = []
                self.times = []

                with tf.name_scope('BPTT'):
                    for i in range(self.BPTT):

                        events_embedded = tf.nn.embedding_lookup(self.Wem,
                                                                 tf.mod(self.events_in[:, i] - 1, self.NUM_CATEGORIES))
                        time = self.times_in[:, i]
                        time_next = self.times_out[:, i]

                        delta_t_prev = tf.expand_dims(time - last_time, axis=-1)
                        delta_t_next = tf.expand_dims(time_next - time, axis=-1)

                        last_time = time

                        time_2d = tf.expand_dims(time, axis=-1)

                        # output, state = RNNcell(events_embedded, state)

                        # TODO Does TF automatically broadcast? Then we'll not
                        # need multiplication with tf.ones
                        type_delta_t = True

                        with tf.name_scope('state_recursion'):
                            # new_state = tf.clip_by_value(
                            #     tf.matmul(state, self.Wh) +
                            #     tf.matmul(events_embedded, self.Wy) +
                            #     # Two ways of interpretting this term
                            #     (tf.matmul(delta_t, self.Wt) if type_delta_t else tf.matmul(time_2d, self.Wt)) +
                            #     tf.matmul(ones_2d, self.bh),
                            #     0.0, 1e9, name='h_t'
                            # )
                            # new_state = tf.minimum(1e9, tf.nn.softplus(
                            #     tf.matmul(state, self.Wh) +
                            #     tf.matmul(events_embedded, self.Wy) +
                            #     # Two ways of interpretting this term
                            #     (tf.matmul(delta_t_prev, self.Wt) if type_delta_t else tf.matmul(time_2d, self.Wt)) +
                            #     tf.matmul(ones_2d, self.bh),
                            #     name='h_t'
                            # ))
                            new_state = tf.tanh(
                                tf.matmul(state, self.Wh) +
                                tf.matmul(events_embedded, self.Wy) +
                                # Two ways of interpretting this term
                                (tf.matmul(delta_t_prev, self.Wt) if type_delta_t else tf.matmul(time_2d, self.Wt)) +
                                tf.matmul(ones_2d, self.bh),
                                name='h_t'
                            )
                            state = tf.where(self.events_in[:, i] > 0, new_state, state)

                        with tf.name_scope('loss_calc'):
                            base_intensity = tf.matmul(ones_2d, self.bt)
                            # wt_non_zero = tf.sign(self.wt) * tf.maximum(1e-9, tf.abs(self.wt))
                            wt_soft_plus = tf.nn.softplus(self.wt)

                            log_lambda_ = (tf.matmul(state, self.Vt) +
                                           (-delta_t_next * wt_soft_plus) +
                                           base_intensity)

                            lambda_ = tf.exp(tf.minimum(50.0, log_lambda_), name='lambda_')

                            log_f_star = (log_lambda_ -
                                          (1.0 / wt_soft_plus) * tf.exp(tf.minimum(50.0, tf.matmul(state, self.Vt) + base_intensity)) +
                                          (1.0 / wt_soft_plus) * lambda_)

                            events_pred = tf.nn.softmax(
                                tf.minimum(50.0,
                                           tf.matmul(state, self.Vy) + ones_2d * self.bk),
                                name='Pr_events'
                            )

                            time_LL = log_f_star
                            mark_LL = tf.expand_dims(
                                tf.log(
                                    tf.maximum(
                                        1e-6,
                                        tf.gather_nd(
                                            events_pred,
                                            tf.concat([
                                                tf.expand_dims(tf.range(self.inf_batch_size), -1),
                                                tf.expand_dims(tf.mod(self.events_out[:, i] - 1, self.NUM_CATEGORIES), -1)
                                            ], axis=1, name='Pr_next_event'
                                            )
                                        )
                                    )
                                ), axis=-1, name='log_Pr_next_event'
                            )
                            step_LL = time_LL + mark_LL

                            # In the batch some of the sequences may have ended before we get to the
                            # end of the seq. In such cases, the events will be zero.
                            # TODO Figure out how to do this with RNNCell, LSTM, etc.
                            num_events = tf.reduce_sum(tf.where(self.events_in[:, i] > 0,
                                                       tf.ones(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE),
                                                       tf.zeros(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE)),
                                                       name='num_events')

                            self.loss -= tf.reduce_sum(
                                tf.where(self.events_in[:, i] > 0,
                                         tf.squeeze(step_LL) / self.batch_num_events,
                                         tf.zeros(shape=(self.inf_batch_size,)))
                            )

                            # self.loss -= tf.cond(num_events > 0,
                            #                      lambda: tf.reduce_sum(
                            #                          tf.where(self.events_in[:, i] > 0,
                            #                                   tf.squeeze(step_LL),
                            #                                   tf.zeros(shape=(self.inf_batch_size,))),
                            #                          name='batch_bptt_loss'),
                            #                      lambda: 0.0)

                        self.time_LLs.append(time_LL)
                        self.mark_LLs.append(mark_LL)
                        self.log_lambdas.append(log_lambda_)

                        self.hidden_states.append(state)
                        self.event_preds.append(events_pred)

                        # self.delta_ts.append(tf.clip_by_value(delta_t, 0.0, np.inf))
                        self.times.append(time)

                self.final_state = self.hidden_states[-1]

                with tf.device(device_cpu):
                    # Global step needs to be on the CPU (Why?)
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.learning_rate = tf.train.inverse_time_decay(self.LEARNING_RATE,
                                                                 global_step=self.global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate)
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

                self.norm_grads, self.global_norm = tf.clip_by_global_norm(grads, 10.0)
                capped_gvs = list(zip(self.norm_grads, vars_))

                with tf.device(device_cpu):
                    tf.contrib.training.add_gradients_summaries(self.gvs)
                    # for g, v in zip(grads, vars_):
                    #     variable_summaries(g, name='grad-' + v.name.split('/')[-1][:-2])

                    variable_summaries(self.loss, name='loss')
                    variable_summaries(self.hidden_states, name='agg-hidden-states')
                    variable_summaries(self.event_preds, name='agg-event-preds-softmax')
                    variable_summaries(self.time_LLs, name='agg-time-LL')
                    variable_summaries(self.mark_LLs, name='agg-mark-LL')
                    variable_summaries(self.time_LLs + self.mark_LLs, name='agg-total-LL')
                    # variable_summaries(self.delta_ts, name='agg-delta-ts')
                    variable_summaries(self.times, name='agg-times')
                    variable_summaries(self.log_lambdas, name='agg-log-lambdas')
                    variable_summaries(tf.nn.softplus(self.wt), name='wt-soft-plus')

                    self.tf_merged_summaries = tf.summary.merge_all()

                self.update = self.optimizer.apply_gradients(capped_gvs,
                                                             global_step=self.global_step)

                self.tf_init = tf.global_variables_initializer()
                # self.check_nan = tf.add_check_numerics_ops()

    def initialize(self, finalize=False):
        """Initialize the global trainable variables."""
        self.sess.run(self.tf_init)

        if finalize:
            # This prevents memory leaks by disallowing changes to the graph
            # after initialization.
            self.sess.graph.finalize()


    def make_feed_dict(self, training_data, batch_idxes, bptt_idx,
                       init_hidden_state=None):
        """Creates a batch for the given batch_idxes starting from bptt_idx.
        The hidden state will be initialized with all zeros if no such state is
        provided.
        """

        if init_hidden_state is None:
            cur_state = np.zeros((len(batch_idxes), self.HIDDEN_LAYER_SIZE))
        else:
            cur_state = init_hidden_state

        train_event_in_seq = training_data['train_event_in_seq']
        train_time_in_seq = training_data['train_time_in_seq']
        train_event_out_seq = training_data['train_event_out_seq']
        train_time_out_seq = training_data['train_time_out_seq']

        batch_event_train_in = train_event_in_seq[batch_idxes, :]
        batch_event_train_out = train_event_out_seq[batch_idxes, :]
        batch_time_train_in = train_time_in_seq[batch_idxes, :]
        batch_time_train_out = train_time_out_seq[batch_idxes, :]

        bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
        bptt_event_in = batch_event_train_in[:, bptt_range]
        bptt_event_out = batch_event_train_out[:, bptt_range]
        bptt_time_in = batch_time_train_in[:, bptt_range]
        bptt_time_out = batch_time_train_out[:, bptt_range]

        if bptt_idx > 0:
            initial_time = batch_time_train_in[:, bptt_idx - 1]
        else:
            initial_time = np.zeros(batch_time_train_in.shape[0])

        feed_dict = {
            self.initial_state: cur_state,
            self.initial_time: initial_time,
            self.events_in: bptt_event_in,
            self.events_out: bptt_event_out,
            self.times_in: bptt_time_in,
            self.times_out: bptt_time_out,
            self.batch_num_events: np.sum(batch_event_train_in > 0)
        }

        return feed_dict

    def train(self, training_data, num_epochs=1,
              restart=False, check_nans=False, one_batch=False,
              with_summaries=False, with_evals=False):
        """Train the model given the training data.

        If with_evals is an integer, then that many elements from the test set
        will be tested.
        """
        create_dir(self.SAVE_DIR)
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)

        # TODO: Should give the variable list explicitly for RMTPP only, in case
        # There are variables outside RMTPP model.
        # TODO: Why does this create new nodes in the graph? Possibly memory leak?
        saver = tf.train.Saver(tf.global_variables())

        if with_summaries:
            train_writer = tf.summary.FileWriter(self.SUMMARY_DIR + '/train',
                                                 self.sess.graph)

        if ckpt and restart:
            print('Restoring from {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        train_event_in_seq = training_data['train_event_in_seq']
        train_time_in_seq = training_data['train_time_in_seq']
        train_event_out_seq = training_data['train_event_out_seq']
        train_time_out_seq = training_data['train_time_out_seq']

        idxes = list(range(len(train_event_in_seq)))
        n_batches = len(idxes) // self.BATCH_SIZE

        for epoch in range(self.last_epoch, self.last_epoch + num_epochs):
            self.rs.shuffle(idxes)

            print("Starting epoch...", epoch)
            total_loss = 0.0

            for batch_idx in range(n_batches):
                # TODO: This is horribly inefficient. Move this to a separate
                # thread using FIFOQueues.
                # However, the previous state from BPTT still needs to be
                # passed to the next BPTT batch. To make this efficient, we
                # will need to set and preserve the previous state in a
                # tf.Variable.
                #  - Sounds like a job for tf.placeholder_with_default?
                #  - Or, of a variable with optinal default?

                batch_idxes = idxes[batch_idx * self.BATCH_SIZE:(batch_idx + 1) * self.BATCH_SIZE]
                batch_event_train_in = train_event_in_seq[batch_idxes, :]
                batch_event_train_out = train_event_out_seq[batch_idxes, :]
                batch_time_train_in = train_time_in_seq[batch_idxes, :]
                batch_time_train_out = train_time_out_seq[batch_idxes, :]

                cur_state = np.zeros((self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE))
                batch_loss = 0.0

                batch_num_events = np.sum(batch_event_train_in > 0)
                for bptt_idx in range(0, len(batch_event_train_in[0]) - self.BPTT, self.BPTT):
                    bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
                    bptt_event_in = batch_event_train_in[:, bptt_range]
                    bptt_event_out = batch_event_train_out[:, bptt_range]
                    bptt_time_in = batch_time_train_in[:, bptt_range]
                    bptt_time_out = batch_time_train_out[:, bptt_range]

                    if np.all(bptt_event_in[:, 0] == 0):
                        # print('Breaking at bptt_idx {} / {}'
                        #       .format(bptt_idx // self.BPTT,
                        #               (len(batch_event_train_in[0]) - self.BPTT) // self.BPTT))
                        break

                    if bptt_idx > 0:
                        initial_time = batch_time_train_in[:, bptt_idx - 1]
                    else:
                        initial_time = np.zeros(batch_time_train_in.shape[0])

                    feed_dict = {
                        self.initial_state: cur_state,
                        self.initial_time: initial_time,
                        self.events_in: bptt_event_in,
                        self.events_out: bptt_event_out,
                        self.times_in: bptt_time_in,
                        self.times_out: bptt_time_out,
                        self.batch_num_events: batch_num_events
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

        if with_evals:
            if isinstance(with_evals, int):
                batch_size = with_evals
            else:
                batch_size = len(training_data['train_event_in_seq'])

            print('Running evaluation on training data: ...')
            train_time_preds, train_event_preds = self.predict_train(training_data,
                                                                     batch_size=batch_size)
            self.eval(train_time_preds[0:batch_size], train_time_out_seq[0:batch_size],
                      train_event_preds[0:batch_size], train_event_out_seq[0:batch_size])

    def restore(self):
        """Restore the model from saved state."""
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)
        print('Loading the model from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, event_in_seq, time_in_seq, single_threaded=False):
        """Treats the entire dataset as a single batch and processes it."""

        all_hidden_states = []
        all_event_preds = []

        cur_state = np.zeros((len(event_in_seq), self.HIDDEN_LAYER_SIZE))

        for bptt_idx in range(0, len(event_in_seq[0]) - self.BPTT, self.BPTT):
            bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
            bptt_event_in = event_in_seq[:, bptt_range]
            bptt_time_in = time_in_seq[:, bptt_range]

            if bptt_idx > 0:
                initial_time = event_in_seq[:, bptt_idx - 1]
            else:
                initial_time = np.zeros(bptt_time_in.shape[0])

            feed_dict = {
                self.initial_state: cur_state,
                self.initial_time: initial_time,
                self.events_in: bptt_event_in,
                self.times_in: bptt_time_in,
            }

            bptt_hidden_states, bptt_events_pred, cur_state = self.sess.run(
                [self.hidden_states, self.event_preds, self.final_state],
                feed_dict=feed_dict
            )

            all_hidden_states.extend(bptt_hidden_states)
            all_event_preds.extend(bptt_events_pred)

        # TODO: This calculation is completely ignoring the clipping which
        # happens during the inference step.
        [Vt, bt, wt]  = self.sess.run([self.Vt, self.bt, self.wt])
        wt = softplus(wt)

        global _quad_worker
        def _quad_worker(params):
            idx, h_i = params
            preds_i = []
            C = np.exp(np.dot(h_i, Vt) + bt).reshape(-1)

            for c_, t_last in zip(C, time_in_seq[:, idx]):
                args = (c_, wt)
                val, _err = quad(quad_func, 0, np.inf, args=args)
                preds_i.append(t_last + val)

            return preds_i

        if single_threaded:
            all_time_preds = [_quad_worker((idx, x)) for idx, x in enumerate(all_hidden_states)]
        else:
            with MP.Pool() as pool:
                all_time_preds = pool.map(_quad_worker, enumerate(all_hidden_states))

        return np.asarray(all_time_preds).T, np.asarray(all_event_preds).swapaxes(0, 1)

    def eval(self, time_preds, time_true, event_preds, event_true):
        """Prints evaluation of the model on the given dataset."""
        # Print test error once every epoch:
        mae, total_valid = MAE(time_preds, time_true, event_true)
        print('** MAE = {:.3f}; valid = {}, ACC = {:.3f}'.format(
            mae, total_valid, ACC(event_preds, event_true)))

    def predict_test(self, data, single_threaded=False):
        """Make (time, event) predictions on the test data."""
        return self.predict(event_in_seq=data['test_event_in_seq'],
                            time_in_seq=data['test_time_in_seq'],
                            single_threaded=single_threaded)

    def predict_train(self, data, single_threaded=False, batch_size=None):
        """Make (time, event) predictions on the training data."""
        if batch_size == None:
            batch_size = data['train_event_in_seq'].shape[0]

        return self.predict(event_in_seq=data['train_event_in_seq'][0:batch_size, :],
                            time_in_seq=data['train_time_in_seq'][0:batch_size, :],
                            single_threaded=single_threaded)
