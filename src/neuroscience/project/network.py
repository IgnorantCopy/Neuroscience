from neuron import *
from optimizer import STDP
from typing import List, Dict, Callable


class Network(Node):
    def __init__(self):
        super().__init__()
        self.injector = []
        self.layers: List[Dict] = []
        self.synapses: List[Synapse] = []
        self.recoder = self.add_recoder(0)
        self.optimizer = self.add_optimizer()
        self.out_spike = []

    def add_current_injector(self, i_fn) -> CurrentInjector:
        current_injector = CurrentInjector(i_fn)
        self.injector.append(current_injector)
        return current_injector

    def add_layer(self, num_neurons: int, **neuron_params) -> NeuronGroup:
        layer = NeuronGroup(num_neurons, **neuron_params)
        layer_info = {
            "group": layer,
            "num": num_neurons,
            "params": neuron_params,
            "neurons": layer.neurons,
        }
        self.layers.append(layer_info)
        return layer

    def add_synapse(self, pre_neuron, post_neuron, **synapse_params) -> Synapse:
        synapse = Synapse(pre_neuron, post_neuron, **synapse_params)
        self.synapses.append(synapse)
        return synapse

    def add_connect(self, layer1, layer2=-1, connect_pattern=None, p=1, **synapse_params) -> List[Synapse]:
        """
        Add synapses between two layers of neurons.
        :param layer1: Index of the first layer
        :param layer2: Index of the second layer
        :param connect_pattern: Connectivity pattern between the two layers
        :param p: Probability of connection
        :param synapse_params: Parameters for the synapses
        :return: The list of synapse
        """
        if layer2 == -1:
            layer2 = layer1 + 1
        group1 = self.layers[layer1]["group"]
        group2 = self.layers[layer2]["group"]
        synapses = group1.connect(group2, connect_pattern, p, **synapse_params)
        for synapse in synapses:
            self.synapses.append(synapse)
        return synapses

    def add_recoder(self, recoder_t) -> Recorder:
        neurons = []
        for layer in self.layers:
            for neuron in layer["neurons"]:
                neurons.append(neuron)
        recoder = Recorder(recoder_t, *neurons)
        self.recoder = recoder
        return recoder

    def add_optimizer(self, tau_pre: float = 2., tau_post: float = 2., f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x) -> STDP:
        optimizer = STDP(self.recoder, self.synapses, tau_pre, tau_post, f_pre, f_post)
        self.optimizer = optimizer
        return optimizer

    def run_network(self):
        """Run the network by running the recoder."""
        self.recoder.run_network(*self.synapses)


    def step(self, dt=0.01):
        nodes = set()
        for synapse in self.synapses:
            nodes.add(synapse.pre_node)
            nodes.add(synapse.post_node)
        for synapse in self.synapses:
            nodes.discard(synapse.pre_node)

        for synapse in self.synapses:
            synapse.pre_node.step(dt)
            synapse.step(dt)
        for node in nodes:
            node.step(dt)
        self.optimizer.step()

    def reset(self):
        for injector in self.injector:
            injector.reset()
        for layer in self.layers:
            for neuron in layer["neurons"]:
                neuron.reset()
        for synapse in self.synapses:
            synapse.reset()
        self.optimizer.reset()
        self.out_spike = np.zeros(len(self.layers[-1]["neurons"]))

    def run(self, t, dt=0.01):
        self.out_spike = np.zeros(len(self.layers[-1]["neurons"]))
        n_t = int(t / dt)
        for i in range(n_t):
            self.step(dt)
        self.optimizer.apply()


