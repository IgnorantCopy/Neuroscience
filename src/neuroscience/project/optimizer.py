from neuron import *
from typing import List, Callable


class STDP:
    def __init__(self, recorder: Recorder, synapses: List[Synapse],
                 tau_pre: float = 2., tau_post: float = 2.,
                 f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x,):
        self.recorder = recorder
        self.synapses = synapses
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.trace_pre = np.zeros(len(self.synapses))
        self.trace_post = np.zeros(len(self.synapses))
        self.delta_w = np.zeros(len(self.synapses))

    def step(self):
        delta_w = np.zeros(len(self.synapses))
        for i, synapse in enumerate(self.synapses):
            weight = synapse.weight
            spike_pre = float(synapse.pre_node.is_spike())
            spike_post = float(synapse.post_node.is_spike())
            self.trace_pre[i] = self.trace_pre[i] - self.trace_pre[i] / self.tau_pre + spike_pre
            self.trace_post[i] = self.trace_post[i] - self.trace_post[i] / self.tau_post + spike_post
            delta_w_pre = -self.f_pre(weight) * self.trace_post[i] * spike_pre
            delta_w_post = self.f_post(weight) * self.trace_pre[i] * spike_post
            delta_w[i] = delta_w_pre + delta_w_post
        self.delta_w += delta_w

    def reset(self):
        self.trace_pre = np.zeros(len(self.synapses))
        self.trace_post = np.zeros(len(self.synapses))
        self.delta_w = np.zeros(len(self.synapses))

    def apply(self):
        for i, synapse in enumerate(self.synapses):
            synapse.weight -= self.delta_w[i]
        self.reset()