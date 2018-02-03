#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: optimizer.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from contextlib import contextmanager
from .gradproc import FilterNoneGrad
from .tower import get_current_tower_context

__all__ = ['apply_grad_processors', 'ProxyOptimizer',
           'PostProcessOptimizer', 'VariableAssignmentOptimizer',
           'AccumGradOptimizer', 'AccumGradOptimizerAlt']


class ProxyOptimizer(tf.train.Optimizer):
    """
    A transparent proxy which delegates all methods of :class:`tf.train.Optimizer`
    """
    def __init__(self, opt, name='ProxyOptimizer'):
        assert isinstance(opt, tf.train.Optimizer), opt
        super(ProxyOptimizer, self).__init__(False, name)
        self._opt = opt

    def compute_gradients(self, *args, **kwargs):
        return self._opt.compute_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._opt.get_slot_names(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        return self._opt.apply_gradients(*args, **kwargs)


def apply_grad_processors(opt, gradprocs):
    """
    Wrapper around optimizers to apply gradient processors.

    Args:
        opt (tf.train.Optimizer):
        gradprocs (list[GradientProcessor]): gradient processors to add to the
            optimizer.

    Returns:
        a :class:`tf.train.Optimizer` instance which runs the gradient
        processors before updating the variables.
    """
    assert isinstance(gradprocs, (list, tuple)), gradprocs

    class _ApplyGradientProcessor(ProxyOptimizer):
        def __init__(self, opt, gradprocs):
            self._gradprocs = gradprocs[:]
            super(_ApplyGradientProcessor, self).__init__(opt)

        def apply_gradients(self, grads_and_vars,
                            global_step=None, name=None):
            g = self._apply(grads_and_vars)
            return self._opt.apply_gradients(g, global_step, name)

        def _apply(self, g):
            for proc in self._gradprocs:
                g = proc.process(g)
            return g

    return _ApplyGradientProcessor(opt, gradprocs)


class PostProcessOptimizer(ProxyOptimizer):
    """
    An optimizer which applies some "post-processing operation" per variable
    (e.g. clipping, quantization) after the gradient update.
    """
    def __init__(self, opt, func, colocate=True):
        """
        Args:
            opt (tf.train.Optimizer):
            func (tf.Variable -> tf.Operation or None): the operation needed
                to perform for this variable after the gradient update.
            colocate (boolean): colocate the function with the variable.
        """
        super(PostProcessOptimizer, self).__init__(opt)
        self._func = func
        self._colocate = colocate

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        update_op = super(PostProcessOptimizer, self).apply_gradients(
            grads_and_vars, global_step)
        ops = []
        with tf.control_dependencies([update_op]):
            for _, var in grads_and_vars:
                with self._maybe_colocate(var):
                    op = self._func(var)
                    if op is not None:
                        assert isinstance(op, tf.Operation), op
                        ops.append(op)
        update_op = tf.group(update_op, *ops, name=name)
        return update_op

    @contextmanager
    def _maybe_colocate(self, var):
        G = tf.get_default_graph()
        if self._colocate:
            with G.colocate_with(var):
                yield
        else:
            yield


class VariableAssignmentOptimizer(PostProcessOptimizer):
    """
    An optimizer which assigns each variable a new value (e.g. clipping,
    quantization) after the gradient update.
    """
    def __init__(self, opt, func):
        """
        Args:
            opt (tf.train.Optimizer):
            func (tf.Variable -> tf.Tensor or None): the new value to be
                assigned to this variable after the gradient update.
        """
        def f(v):
            t = func(v)
            if t is None:
                return t
            return tf.assign(v, t, use_locking=False).op
        super(VariableAssignmentOptimizer, self).__init__(opt, f)

class AccumGradOptimizerAlt(ProxyOptimizer):
    """
    An optimizer which accumulates gradients across :
    All operator between compute_gradients and apply_gradients must
    apply follow function: run_or_not(ok, no)
    ex: 
    grads = allreduce_grads(accum_grad_list)
    =>
    with tf.control_dependencies([accum_grad_list]):
        grads = run_or_not(allreduce_grads(accum_grad_list), accum_grad_list[0])
    
    """

    def __init__(self, opt, niter=1):
        """
        Args:
            opt (tf.train.Optimizer): the underlying sub-optimizer.
            niter (int): number of iterations to accumulate gradients.
        """
        super(AccumGradOptimizerAlt, self).__init__(opt, 'AccumGrad')
        self._niter = int(niter)

        with tf.variable_scope(tf.get_variable_scope()):
            self._counter = tf.Variable(0, name="counter", trainable=False, dtype=tf.int32)
        # ==================================
        # update counter lambda
        def counter_add():
          return tf.assign_add(self._counter, 1)
        def counter_reset():
          return tf.assign(self._counter, 1)
        # ==================================
        self._update_counter = tf.cond(tf.equal(self._counter, self._niter), counter_reset, counter_add, name='update_counter')
        with tf.control_dependencies([self._update_counter]):
            self._pred = tf.equal(self._counter, self._niter)

    def _create_accum_slots(self, var_list):
        org = tf.get_variable_scope().reuse
        tf.get_variable_scope()._reuse = tf.AUTO_REUSE

        slots = [self._zeros_slot(v, "accum_grad", self._name) for v in  var_list]

        tf.get_variable_scope()._reuse = org
        return slots

    def compute_gradients(self, *args, **kwargs):
        # get trainalbe variable
        if get_current_tower_context()!=None and get_current_tower_context().has_own_variables:
            trainable_var = get_current_tower_context().get_collection_in_tower(
                tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            trainable_var = tf.trainable_variables()

        # Another method to get trainable variable

        #from tensorflow.python.framework import ops
        #from tensorflow.python.util import nest
        #from tensorflow.python.eager import context
        #from tensorflow.python.ops import variables

        #if context.in_eager_mode():
        #    raise RuntimeError("accum not support eager mode")
        #if(kwargs.get("var_list") != None):
        #    trainable_var = nest.flatten(kwargs.get("var_list"))
        #else:
        #    trainable_var = (
        #        variables.trainable_variables() +
        #        ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        #    raise RuntimeError("var_list can't be empty")
        #trainable_var += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)

        # slots
        self._accum_slots = self._create_accum_slots(trainable_var)

        # ==================================
        # clear grads lambda
        def grads_clear():
            clear_ops = [tf.assign(s, tf.zeros_like(s)) for s in self._accum_slots]
            return tf.group(*clear_ops, name='clear_grads')
        # ===================================

        with tf.control_dependencies([self._update_counter]):
            cond_clear_grads = tf.cond(tf.equal(self._counter, 1), grads_clear, tf.no_op, name='cond_clear_grads')

        grads_and_vars = self._opt.compute_gradients(*args, **kwargs)
  
        self._grads = [g for g, _ in grads_and_vars]

        with tf.control_dependencies([cond_clear_grads]):
            self._accum_grads = [tf.assign_add(s, tf.divide(g, self._niter)) for s, (g, _) in zip(self._accum_slots, grads_and_vars)]
            return zip(self._accum_grads, trainable_var)

    def apply_gradients(self, *args, **kwargs):
        def update_grad():
            update_op = self._opt.apply_gradients(*args, **kwargs)
            return update_op

        return tf.cond(self._pred, update_grad, tf.no_op, name='cond_apply_gradients')


class AccumGradOptimizer(ProxyOptimizer):
    """
    An optimizer which accumulates gradients across :math:`k` :meth:`minimize` calls,
    and apply them together in every :math:`k`th :meth:`minimize` call.
    This is equivalent to using a :math:`k` times larger batch size plus a
    :math:`k` times larger learning rate, but uses much less memory.

    Note that this implementation may not support all models.
    E.g., it doesn't support sparse gradient update.
    """

    def __init__(self, opt, niter):
        """
        Args:
            opt (tf.train.Optimizer): the underlying sub-optimizer.
            niter (int): number of iterations to accumulate gradients.
        """
        super(AccumGradOptimizer, self).__init__(opt, 'AccumGrad')
        self._niter = int(niter)

    def _create_accum_slots(self, var_list):
        slots = []
        for v in var_list:
            # TODO an option to not colocate the accumulators with variables (to save more memory)
            s = self._zeros_slot(v, "accum", self._name)
            slots.append(s)
        return slots

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        #assert global_step is None, \
        #    "AccumGradOptimizer doesn't support the option global_step! " \
        #    "Please maintain it yourself."
        grads_and_vars = FilterNoneGrad().process(grads_and_vars)
        vs = []
        for g, v in grads_and_vars:
            assert isinstance(g, tf.Tensor) and isinstance(v, tf.Variable), \
                "AccumGradOptimizer only works for dense update! " \
                "Types of v and g are {} and {}".format(type(v), type(g))
            vs.append(v)

        with tf.control_dependencies(None):
            slots = self._create_accum_slots(vs)
            slots_and_vars = [(s, gv[1]) for s, gv in zip(slots, grads_and_vars)]

            # Create the counter on the same device as the first variable.
            with tf.variable_scope(self._name), \
                    vs[0].graph.colocate_with(vs[0]):
                counter = tf.Variable(
                    0, name="counter", trainable=False, dtype=tf.int32)

        with tf.name_scope('AccumGradOptimizer'):
            ops = []
            for s, gv in zip(slots, grads_and_vars):
                g, v = gv
                ops.append(s.assign_add(g))

            def counter_add():
              return tf.assign_add(counter, 1)
            def counter_reset():
              return tf.assign(counter, 0)

            #update_counter = tf.assign_add(counter, 1, name='update_counter')
            update_counter = tf.cond(tf.equal(counter, self._niter - 1), counter_reset, counter_add)
            update_slot_op = tf.group(update_counter, *ops, name='update_slot')

            def update_grad():
                update_op = self._opt.apply_gradients(slots_and_vars, global_step)
                with tf.control_dependencies([update_op]):
                    clear_ops = [tf.assign(s, tf.zeros_like(s)) for s in slots]
                return tf.group(*clear_ops, name='update_grad')

            #pred = tf.equal(tf.mod(counter, self._niter), 0)
            pred = tf.equal(counter, self._niter - 1)
            with tf.control_dependencies([update_slot_op]):
                if name is None:
                    name = 'cond_update_grad'
                op = tf.cond(pred, update_grad, tf.no_op, name=name).op
        return op


if __name__ == '__main__':
    # run it with "python -m tensorpack.tfutils.optimizer"

    x = tf.get_variable('x', shape=[6])
    cost = tf.reduce_sum(tf.abs(x), name='cost')
    opt = tf.train.GradientDescentOptimizer(0.01)
    opt = AccumGradOptimizer(opt, 5)
    min_op = opt.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        for k in range(20):
            min_op.run()
            print(x.eval())
