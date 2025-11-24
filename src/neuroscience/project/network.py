from neuron import *
from typing import List, Dict


class Network:
    def __init__(self):
        self.injector = []
        self.layers: List[Dict]      = []
        self.synapses: List[Synapse] = []
        self.recoder = None

    def add_current_injector(self, i_fn):
        """
        Constructor for CurrentInjector class.
        :param i_fn: function that returns the current (mA) as a function of time (ms)
        :return: CurrentInjector object
        """
        current_injector = CurrentInjector(i_fn)
        self.injector.append(current_injector)
        return current_injector

    def add_layer(self, num_neurons: int, **neuron_params):
        """
        Add a layer of neurons to the network.
        :param num_neurons: Number of neurons in the layer
        :param neuron_params: Parameters for the neurons in the layer
        :return: The neuron group
        """
        layer = NeuronGroup(num_neurons, **neuron_params)
        layer_info = {
            "group": layer,
            "num": num_neurons,
            "params": neuron_params,
            "neurons": layer.neurons,
        }
        self.layers.append(layer_info)
        return layer

    def add_synapse(self, pre_neuron, post_neuron, **synapse_params):
        """
        Add a synapse between two neurons.
        :param pre_neuron: Pre-synaptic neuron
        :param post_neuron: Post-synaptic neuron
        :param synapse_params: Parameters for synapse
        :return: The synapse
        """
        synapse = Synapse(pre_neuron, post_neuron, **synapse_params)
        self.synapses.append(synapse)
        return synapse

    def add_connect(self, layer1, layer2=-1, connect_pattern=None, p=1, **synapse_params):
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

    def add_recoder(self, recoder_t):
        neurons = []
        for layer in self.layers:
            for neuron in layer["neurons"]:
                neurons.append(neuron)
        recoder = Recorder(recoder_t, *neurons)
        self.recoder = recoder
        return recoder

    def run(self):
        self.recoder.run_network(*self.synapses)
