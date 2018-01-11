# Sample Test passing with nose and pytest

# This assumes that the system installed version of tf_rmtpp is the development
# build.
# TODO: Is there a way of importing the active code without having to run:
# `python setup.py develop`?
import tf_rmtpp
import numpy as np
import decorated_options as Deco
import tensorflow as tf


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)[:, None]


def map_feed_dict_to_np(f_d, rmtpp_mdl):
    return {
        'events_in': f_d[rmtpp_mdl.events_in],
        'events_out': f_d[rmtpp_mdl.events_out],
        'initial_state': f_d[rmtpp_mdl.initial_state],
        'initial_time': f_d[rmtpp_mdl.initial_time],
        'times_in': f_d[rmtpp_mdl.times_in],
        'times_out': f_d[rmtpp_mdl.times_out],
        'batch_num_events': f_d[rmtpp_mdl.batch_num_events]
    }


@Deco.optioned()
def numpy_LL(batch, num_categories,
             bptt, Wt, Wem, Wh, bh, wt, Wy, Vy, Vt, bt, bk):
    """Calculate the LL of the given data for 1 BPTT in the batch by replicating the
    computation graph calculations."""

    events_in = batch['events_in']
    times_in = batch['times_in']
    events_out = batch['events_out']
    times_out = batch['times_out']
    batch_events = batch['batch_num_events']

    last_time = batch['initial_time']
    state = batch['initial_state']

    Wem_ = Wem(num_categories)
    Vy_ = Vy(num_categories)
    bk_ = bk(num_categories)

    loss = 0.0
    time_LLs = []
    mark_LLs = []
    hidden_states = []
    log_lambdas = []
    event_preds = []

    for i in range(bptt):
        events_embedded = Wem_[events_in[:, i] - 1, :]
        time = times_in[:, i]
        time_next = times_out[:, i]

        delta_t_prev = (time - last_time)[:, None]
        delta_t_next = (time_next - time)[:, None]

        last_time = time

        time_2d = time[:, None]

        type_delta_t = True

        # new_state = np.minimum(
        #     1e9,
        #     tf_rmtpp.rmtpp_core.softplus(
        #         (state @ Wh) +
        #         (events_embedded @ Wy) +
        #         ((delta_t_prev @ Wt) if type_delta_t else (time_2d @ Wt)) +
        #         bh
        #     )
        # )
        new_state = np.tanh(
            (state @ Wh) +
            (events_embedded @ Wy) +
            ((delta_t_prev @ Wt) if type_delta_t else (time_2d @ Wt)) +
            bh
        )

        state = np.where(events_in[:, i][:, None] > 0, new_state, state)

        wt_soft_plus = tf_rmtpp.rmtpp_core.softplus(wt)

        log_lambda_ = (state @ Vt) + bt + (-delta_t_next * wt_soft_plus)

        lambda_ = np.exp(np.minimum(50, log_lambda_))

        log_f_star = (log_lambda_ -
                      (1.0 / wt_soft_plus) * np.exp(np.minimum(50.0, (state @ Vt) + bt)) +
                      (1.0 / wt_soft_plus) * lambda_)

        events_pred = softmax(np.minimum(50, (state @ Vy_) + bk_))

        time_LL = log_f_star
        mark_LL = np.log(np.maximum(
            1e-6,
            events_pred[np.arange(events_pred.shape[0]), events_out[:, i] - 1]
        ))[:, None]

        step_LL = time_LL + mark_LL

        num_events = np.sum(events_in[:, i] > 0)

        loss -= np.sum((step_LL / batch_events)[events_in[:, i] > 0])

        time_LLs.append(time_LL)
        mark_LLs.append(mark_LL)
        hidden_states.append(state)
        log_lambdas.append(log_lambda_)
        event_preds.append(events_pred)

    return {
        'loss': loss,
        'time_LLs': time_LLs,
        'mark_LLs': mark_LLs,
        'hidden_states': hidden_states,
        'log_lambdas': log_lambdas,
        'event_preds': event_preds
    }


def dummy_data():
    return {
        'num_categories': 22,
        'train_event_in_seq': np.array([[ 9,  9,  4,  4,  4,  4,  5,  4,  4, 10,  1,  9,  4,  9,  4,  9,  8,
                 8, 18,  7],
               [ 4,  4,  4,  9, 10,  9, 12,  1,  6,  4,  8, 18,  8,  4,  4,  9,  4,
                10, 11,  4]], dtype=np.int32),
        'train_event_out_seq': np.array([[ 9,  4,  4,  4,  4,  5,  4,  4, 10,  1,  9,  4,  9,  4,  9,  8,  8,
                18,  7,  4],
               [ 4,  4,  9, 10,  9, 12,  1,  6,  4,  8, 18,  8,  4,  4,  9,  4, 10,
                11,  4,  9]], dtype=np.int32),
        'train_time_in_seq': np.array([[ 0.03780895,  0.05659572,  0.0711524 ,  0.09084344,  0.1167087 ,
                 0.11749727,  0.11836644,  0.12978144,  0.14427023,  0.14588072,
                 0.17034163,  0.17376789,  0.19021592,  0.21238699,  0.21258176,
                 0.21540879,  0.21883103,  0.21883143,  0.21883258,  0.22609817],
               [ 0.02668075,  0.03325118,  0.03622794,  0.08866453,  0.09788354,
                 0.0990687 ,  0.18555974,  0.1870842 ,  0.21152709,  0.21570823,
                 0.21883037,  0.21883269,  0.22277321,  0.22378501,  0.23709134,
                 0.26201022,  0.2624901 ,  0.27474064,  0.29912653,  0.30328643]]),
        'train_time_out_seq': np.array([[ 0.05659572,  0.0711524 ,  0.09084344,  0.1167087 ,  0.11749727,
                 0.11836644,  0.12978144,  0.14427023,  0.14588072,  0.17034163,
                 0.17376789,  0.19021592,  0.21238699,  0.21258176,  0.21540879,
                 0.21883103,  0.21883143,  0.21883258,  0.22609817,  0.25679366],
               [ 0.03325118,  0.03622794,  0.08866453,  0.09788354,  0.0990687 ,
                 0.18555974,  0.1870842 ,  0.21152709,  0.21570823,  0.21883037,
                 0.21883269,  0.22277321,  0.22378501,  0.23709134,  0.26201022,
                 0.2624901 ,  0.27474064,  0.29912653,  0.30328643,  0.31445738]])
    }


def dummy_data_with_missing():
    return {
        'num_categories': 22,
        'train_event_in_seq': np.array([[ 9,  9,  4,  4,  4,  4,  5,  4,  4, 10,  1,  9,  0,  0,  0,  0,  0,
                 0, 0, 0],
               [ 4,  4,  4,  9, 10,  9, 12,  1,  6,  4,  8, 18,  8,  4,  4,  9,  4,
                10, 11,  4]], dtype=np.int32),
        'train_event_out_seq': np.array([[ 9,  4,  4,  4,  4,  5,  4,  4, 10,  1,  9,  4,  9,  4,  9,  8,  8,
                18,  7,  4],
               [ 4,  4,  9, 10,  9, 12,  1,  6,  4,  8, 18,  8,  4,  4,  9,  4, 10,
                11,  4,  9]], dtype=np.int32),
        'train_time_in_seq': np.array([[ 0.03780895,  0.05659572,  0.0711524 ,  0.09084344,  0.1167087 ,
                 0.11749727,  0.11836644,  0.12978144,  0.14427023,  0.14588072,
                 0.17034163,  0.17376789,  0.19021592,  0.21238699,  0.21258176,
                 0.21540879,  0.21883103,  0.21883143,  0.21883258,  0.22609817],
               [ 0.02668075,  0.03325118,  0.03622794,  0.08866453,  0.09788354,
                 0.0990687 ,  0.18555974,  0.1870842 ,  0.21152709,  0.21570823,
                 0.21883037,  0.21883269,  0.22277321,  0.22378501,  0.23709134,
                 0.26201022,  0.2624901 ,  0.27474064,  0.29912653,  0.30328643]]),
        'train_time_out_seq': np.array([[ 0.05659572,  0.0711524 ,  0.09084344,  0.1167087 ,  0.11749727,
                 0.11836644,  0.12978144,  0.14427023,  0.14588072,  0.17034163,
                 0.17376789,  0.19021592,  0.21238699,  0.21258176,  0.21540879,
                 0.21883103,  0.21883143,  0.21883258,  0.22609817,  0.25679366],
               [ 0.03325118,  0.03622794,  0.08866453,  0.09788354,  0.0990687 ,
                 0.18555974,  0.1870842 ,  0.21152709,  0.21570823,  0.21883037,
                 0.21883269,  0.22277321,  0.22378501,  0.23709134,  0.26201022,
                 0.2624901 ,  0.27474064,  0.29912653,  0.30328643,  0.31445738]])
    }


def test_LL():
    # Step 1: fake data
    data = dummy_data()

    tf.reset_default_graph()
    with tf.Session() as sess:
        rmtpp_mdl = tf_rmtpp.rmtpp_core.RMTPP(
            sess=sess,
            num_categories=data['num_categories'],
            summary_dir='/tmp',
            _opts=tf_rmtpp.def_opts
        )

        rmtpp_mdl.initialize(finalize=False)

        f_d = rmtpp_mdl.make_feed_dict(data, [0, 1], 0)
        np_batch = map_feed_dict_to_np(f_d, rmtpp_mdl=rmtpp_mdl)

        tf_vals = rmtpp_mdl.sess.run(
            {
                'loss': rmtpp_mdl.loss,
                'mark_LLs': rmtpp_mdl.mark_LLs,
                'time_LLs': rmtpp_mdl.time_LLs,
                'hidden_states': rmtpp_mdl.hidden_states,
                'log_lambdas': rmtpp_mdl.log_lambdas,
                'event_preds': rmtpp_mdl.event_preds
            },
            feed_dict=f_d
        )

        np_vals = numpy_LL(
            np_batch,
            num_categories=data['num_categories'],
            _opts=tf_rmtpp.def_opts
        )

    assert np.abs((tf_vals['loss'] - np_vals['loss']) / tf_vals['loss']) < 0.01


def test_LL_with_missing():
    # Step 1: fake data
    data = dummy_data_with_missing()

    tf.reset_default_graph()
    with tf.Session() as sess:
        rmtpp_mdl = tf_rmtpp.rmtpp_core.RMTPP(
            sess=sess,
            num_categories=data['num_categories'],
            summary_dir='/tmp',
            _opts=tf_rmtpp.def_opts
        )

        rmtpp_mdl.initialize(finalize=False)

        f_d = rmtpp_mdl.make_feed_dict(data, [0, 1], 0)
        np_batch = map_feed_dict_to_np(f_d, rmtpp_mdl=rmtpp_mdl)

        tf_vals = rmtpp_mdl.sess.run(
            {
                'loss': rmtpp_mdl.loss,
                'mark_LLs': rmtpp_mdl.mark_LLs,
                'time_LLs': rmtpp_mdl.time_LLs,
                'hidden_states': rmtpp_mdl.hidden_states,
                'log_lambdas': rmtpp_mdl.log_lambdas,
                'event_preds': rmtpp_mdl.event_preds
            },
            feed_dict=f_d
        )

        np_vals = numpy_LL(
            np_batch,
            num_categories=data['num_categories'],
            _opts=tf_rmtpp.def_opts
        )


    assert np.abs((tf_vals['loss'] - np_vals['loss']) / tf_vals['loss']) < 0.01


def test_pass():
    assert True, "dummy sample test"
