from neuron import *
from typing import List, Callable


# class STDP:
#     def __init__(self, synapse_groups: List[SynapseGroup],
#                  tau_pre: float = 2., tau_post: float = 2.,
#                  f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x):
#         self.synapse_groups = synapse_groups
#         self.tau_pre = tau_pre
#         self.tau_post = tau_post
#         self.f_pre = f_pre
#         self.f_post = f_post
#         self.trace_pre = [np.zeros(len(synapse_group.pre_group.neurons)) for synapse_group in self.synapse_groups]
#         self.trace_post = [np.zeros(len(synapse_group.post_group.neurons)) for synapse_group in self.synapse_groups]
#         self.delta_w = [np.zeros((len(synapse_group.pre_group.neurons), len(synapse_group.post_group.neurons)))
#                         for synapse_group in self.synapse_groups]
#
#     def step(self):
#         for i, synapse_group in enumerate(self.synapse_groups):
#             n_in = len(synapse_group.pre_group.neurons)
#             n_out = len(synapse_group.post_group.neurons)
#             weight = np.array([synapse.weight for synapse in synapse_group.synapses]).reshape((n_in, n_out))
#             spike_pre = np.array([float(neuron.is_spike()) for neuron in synapse_group.pre_group.neurons])
#             spike_post = np.array([float(neuron.is_spike()) for neuron in synapse_group.post_group.neurons])
#
#             self.trace_pre[i] = self.trace_pre[i] - self.trace_pre[i] / self.tau_pre + spike_pre
#             self.trace_post[i] = self.trace_post[i] - self.trace_post[i] / self.tau_post + spike_post
#             delta_w_pre = -self.f_pre(weight) * (self.trace_post[i][None, :] * spike_pre[:, None])
#             delta_w_post = self.f_post(weight) * (self.trace_pre[i][:, None] * spike_post[None, :])
#             self.delta_w[i] += delta_w_pre + delta_w_post
#
#     def reset(self):
#         self.trace_pre = [np.zeros(len(synapse_group.pre_group.neurons)) for synapse_group in self.synapse_groups]
#         self.trace_post = [np.zeros(len(synapse_group.post_group.neurons)) for synapse_group in self.synapse_groups]
#         self.delta_w = [np.zeros((len(synapse_group.pre_group.neurons), len(synapse_group.post_group.neurons)))
#                         for synapse_group in self.synapse_groups]
#
#     def apply(self):
#         for i, synapse_group in enumerate(self.synapse_groups):
#             n_out = len(synapse_group.post_group.neurons)
#             for j, synapse in enumerate(synapse_group.synapses):
#                 synapse.weight -= self.delta_w[i][j // n_out, j % n_out]
#         self.reset()


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