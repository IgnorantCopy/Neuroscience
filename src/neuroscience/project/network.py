from neuron import *
from typing import List


class Network(Node):
    def __init__(self, layers: List[Union[Neuron, NeuronGroup]], **synapse_params):
        super().__init__()
        self.injectors = []
        self.layers = []
        self.synapses = []
        self.in_synapses = []
        self.recoder = self.update_recoder(0)
        self.make_layers(layers, synapse_params)
        self.out_spike = np.zeros(len(self.layers[-1].neurons))

    def add_current_injector(self, i_fn, **synapses_params):
        current_injector = CurrentInjector(i_fn)
        self.injectors.append(current_injector)
        synapses = SynapseGroup(current_injector, self.layers[0], **synapses_params)
        self.in_synapses.append(synapses)

    def make_layers(self, layers: List[Union[Neuron, NeuronGroup]], synapse_params):
        self.add_layer(layers[0])
        for i in range(1, len(layers)):
            self.add_layer(layers[i])
            self.add_synapse(self.layers[-2], self.layers[-1], synapse_params)

    def add_layer(self, layer: Union[Neuron, NeuronGroup]):
        if isinstance(layer, Neuron):
            self.layers.append(NeuronGroup(1, **layer.hyper_parameters()))
        elif isinstance(layer, NeuronGroup):
            self.layers.append(layer)
        else:
            raise ValueError(f"Invalid layer type {type(layer)}")

    def add_synapse(self, pre_neuron, post_neuron, synapse_params):
        weight = synapse_params.get("weight", None)
        if weight is not None:
            weight_pattern = lambda i, j: weight
        else:
            weight_pattern = synapse_params.get("weight_pattern", lambda i, j: np.clip(np.random.randn(), -1.5, 1.5))
        synapse = SynapseGroup(pre_neuron, post_neuron, weight_pattern=weight_pattern, **synapse_params)
        self.synapses.append(synapse)

    def update_synapse(self, index, connect_pattern=None, weight_pattern=None, p=1, **synapse_params):
        original_params = self.synapses[index].hyper_parameters()
        original_params.update(synapse_params)
        connect_pattern = connect_pattern or self.synapses[index].connect_pattern
        weight_pattern = weight_pattern or self.synapses[index].weight_pattern
        self.synapses[index].update_params(connect_pattern=connect_pattern, weight_pattern=weight_pattern, p=p, **original_params)

    def update_in_synapse(self, index, connect_pattern=None, weight_pattern=None, p=1, **synapse_params):
        original_params = self.in_synapses[index].hyper_parameters()
        original_params.update(synapse_params)
        connect_pattern = connect_pattern or self.in_synapses[index].connect_pattern
        weight_pattern = weight_pattern or self.in_synapses[index].weight_pattern
        self.in_synapses[index].update_params(connect_pattern=connect_pattern, weight_pattern=weight_pattern, p=p, **original_params)

    def update_recoder(self, recoder_t):
        neurons = []
        for layer in self.layers:
            neurons.extend(layer.neurons)
        self.recoder = Recorder(recoder_t, *neurons)

    def step(self, dt=0.01):
        self.t += dt
        self.recoder.t += 1
        for synapses in self.in_synapses:
            synapses.step(dt)
        for synapses in self.synapses:
            synapses.step(dt)
            for neuron in synapses.pre_group.neurons:
                self.recoder.update(neuron)
        for i, node in enumerate(self.layers[-1].neurons):
            node.step(dt)
            self.recoder.update(node)
            self.out_spike[i] += node.is_spike()

    def reset(self):
        for injector in self.injectors:
            injector.reset()
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.reset()
        for synapse in self.synapses:
            synapse.reset()
        for synapse in self.in_synapses:
            synapse.reset()

    def run(self, t, dt=0.01):
        self.update_recoder(t)
        n_t = int(t / dt)
        for i in range(n_t):
            self.step(dt)

    def save_weights(self, path):
        for i, synapse_group in enumerate(self.synapses):
            weights = np.array([s.weight for s in synapse_group.synapses])
            np.save(f"{path}/synapses_{i}.npy", weights)

    def load_weights(self, path):
        for i, synapse_group in enumerate(self.synapses):
            weights = np.load(f"{path}/synapses_{i}.npy")
            for j, s in enumerate(synapse_group.synapses):
                s.weight = weights[j]
