from neuron import *
from typing import List, Callable


class STDP(Node):
    def __init__(self, synapse_groups: List[SynapseGroup],
                 tau_pos: float = 10., tau_neg: float = 10.,
                 f_pos: Callable = lambda x: x, f_neg: Callable = lambda x: x,):
        super().__init__()
        self.synapse_groups = synapse_groups
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.f_pos = f_pos
        self.f_neg = f_neg
        self.spikes_pre = [
            [[] for _ in range(len(synapse_group.pre_group.neurons))]
            for synapse_group in self.synapse_groups
        ]
        self.spikes_post = [
            [[] for _ in range(len(synapse_group.post_group.neurons))]
            for synapse_group in self.synapse_groups
        ]

    def step(self, dt=0.01):
        self.t += dt
        for i, synapse_group in enumerate(self.synapse_groups):
            for j, neuron in enumerate(synapse_group.pre_group.neurons):
                if neuron.is_spike():
                    self.spikes_pre[i][j].append(self.t)
            for j, neuron in enumerate(synapse_group.post_group.neurons):
                if neuron.is_spike():
                    self.spikes_post[i][j].append(self.t)

    def reset(self):
        self.spikes_pre = [
            [[] for _ in range(len(synapse_group.pre_group.neurons))]
            for synapse_group in self.synapse_groups
        ]
        self.spikes_post = [
            [[] for _ in range(len(synapse_group.post_group.neurons))]
            for synapse_group in self.synapse_groups
        ]

    def w_fn(self, weight, delta_t):
        if delta_t > 0:
            return self.f_pos(weight) * np.exp(-delta_t / self.tau_pos)
        elif delta_t < 0:
            return -self.f_neg(weight) * np.exp(delta_t / self.tau_neg)
        else:
            return 0.

    def apply(self):
        for i in range(len(self.synapse_groups)):
            spikes_pre = self.spikes_pre[i]
            spikes_post = self.spikes_post[i]
            for j in range(len(spikes_pre)):
                for k in range(len(spikes_post)):
                    spike_pre = spikes_pre[j]       # list of spike times for pre-synaptic neuron
                    spike_post = spikes_post[k]     # list of spike times for post-synaptic neuron
                    dw = 0.
                    index = j * len(spikes_post) + k
                    synapse = self.synapse_groups[i].synapses[index]
                    for t_pre in spike_pre:
                        for t_post in spike_post:
                            delta_t = t_post - t_pre
                            dw += self.w_fn(synapse.weight, delta_t)
                    synapse.weight -= dw
                    synapse.weight = np.clip(synapse.weight, -1.5, 1.5)
        self.reset()