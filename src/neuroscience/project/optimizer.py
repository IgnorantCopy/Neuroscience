from neuron import *
from typing import List, Callable


class STDP:
    def __init__(self, synapse_groups: List[SynapseGroup],
                 tau_pre: float = 2., tau_post: float = 2.,
                 f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x):
        self.synapse_groups = synapse_groups
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.trace_pre = [np.zeros(len(synapse_group.synapses)) for synapse_group in self.synapse_groups]
        self.trace_post = [np.zeros(len(synapse_group.synapses)) for synapse_group in self.synapse_groups]
        self.delta_w = [np.zeros(len(synapse_group.synapses)) for synapse_group in self.synapse_groups]

    def step(self):
        delta_w = [np.zeros(len(synapse_group.synapses)) for synapse_group in self.synapse_groups]
        for i, synapse_group in enumerate(self.synapse_groups):
            for j, synapse in enumerate(synapse_group.synapses):
                weight = synapse.weight
                spike_pre = float(synapse.pre_node.is_spike())
                spike_post = float(synapse.post_node.is_spike())
                self.trace_pre[i][j] -= self.trace_pre[i][j] / self.tau_pre + spike_pre
                self.trace_post[i][j] -= self.trace_post[i][j] / self.tau_post + spike_post
                delta_w_pre = -self.f_pre(weight) * self.trace_post[i][j] * spike_pre
                delta_w_post = self.f_post(weight) * self.trace_pre[i][j] * spike_post
                delta_w[i][j] = delta_w_pre + delta_w_post
        self.delta_w += delta_w

    def reset(self):
        self.trace_pre = [np.zeros(len(synapse_group.synapses)) for synapse_group in self.synapse_groups]
        self.trace_post = [np.zeros(len(synapse_group.synapses)) for synapse_group in self.synapse_groups]
        self.delta_w = [np.zeros(len(synapse_group.synapses)) for synapse_group in self.synapse_groups]

    def apply(self):
        for i, synapse_group in enumerate(self.synapse_groups):
            for j, synapse in enumerate(synapse_group.synapses):
                synapse.weight -= self.delta_w[i][j]
        self.reset()