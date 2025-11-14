import numpy as np
from tqdm import tqdm
from abc import abstractmethod, ABC


class Node(ABC):
    def __init__(self):
        self.t = 0.0

    @abstractmethod
    def step(self, dt=0.01):
        pass

    @abstractmethod
    def reset(self):
        pass

    def run(self, t, dt=0.01):
        """
        run the neuron for t (ms)
        :param t: total time to run (ms)
        :param dt: time step (ms)
        :return: None
        """
        n_t = int(t / dt)
        for i in range(n_t):
            self.step(dt)


class Neuron(Node):
    def __init__(self, c_m=1.0, g_l=0.3, g_k=36.0, g_na=120.0, n=0.32, m=0.05, h=0.6, e_l=-54.387, e_k=-77.0,
                 e_na=50.0, d=10.0, l=1000.0, r_a=100.0):
        """
        Constructor for Neuron class.
        :param c_m: membrane conductance (μF/cm^2)
        :param g_l: leak conductance (S/cm^2)
        :param g_k: potassium conductance (S/cm^2)
        :param g_na: sodium conductance (S/cm^2)
        :param n: probability of K activation gates are open
        :param m: probability of Na activation gates are open
        :param h: probability of Na inactivation gates are open
        :param e_l: leak reversal potential (mV)
        :param e_k: potassium reversal potential (mV)
        :param e_na: sodium reversal potential (mV)
        :param d: cable diameter (μm)
        :param l: cable length (μm)
        :param r_a: axial resistance (Ω*cm)
        """
        super().__init__()
        self.c_m = c_m
        self.g_l = g_l
        self.g_k = g_k
        self.g_na = g_na
        self.init_n = n
        self.init_m = m
        self.init_h = h
        self.e_l = e_l
        self.e_k = e_k
        self.e_na = e_na
        self.d = d
        self.l = l
        self.r_a = r_a

        self.dx = 20.0  # spatial step (μm)
        self.n_x = int(l / self.dx) + 1  # number of spatial steps
        self.i_inj = []  # input current (mA)
        self.current = 0.0
        self.v = -65.0 * np.ones(self.n_x)  # membrane potential (mV)
        self.n = self.init_n * np.ones(self.n_x)
        self.m = self.init_m * np.ones(self.n_x)
        self.h = self.init_h * np.ones(self.n_x)

    def step(self, dt=0.01):
        self.t += dt

        d2v_dx2 = np.ones(self.n_x, dtype=np.float64)
        d2v_dx2[0] = (self.v[1] - 2 * self.v[0] + self.v[0]) / self.dx ** 2
        d2v_dx2[-1] = (self.v[-2] - 2 * self.v[-1] + self.v[-1]) / self.dx ** 2
        d2v_dx2[1:-1] = (self.v[2:] - 2 * self.v[1:-1] + self.v[:-2]) / self.dx ** 2

        alpha_m = (0.1 * (self.v + 40.0)) / (1.0 - np.exp(-(self.v + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(self.v + 65.0) / 18.0)
        alpha_h = 0.07 * np.exp(-(self.v + 65.0) / 20.0)
        beta_h = 1.0 / (np.exp(-(self.v + 35.0) / 10.0) + 1.0)
        alpha_n = 0.01 * (self.v + 55.0) / (1.0 - np.exp(-(self.v + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(self.v + 65.0) / 80.0)

        self.n = self.n + dt * (alpha_n * (1.0 - self.n) - beta_n * self.n)
        self.m = self.m + dt * (alpha_m * (1.0 - self.m) - beta_m * self.m)
        self.h = self.h + dt * (alpha_h * (1.0 - self.h) - beta_h * self.h)

        self.current = sum(self.i_inj)
        self.i_inj = []
        i_inj = np.array([self.current] + [0.0] * (self.n_x - 1))

        self.v = self.v + dt / self.c_m * (d2v_dx2 * (self.d * 1e4) / (4 * self.r_a)
                                           - self.g_l * (self.v - self.e_l)
                                           - self.g_k * (self.v - self.e_k) * (self.n ** 4)
                                           - self.g_na * (self.v - self.e_na) * (self.m ** 3) * self.h
                                           + i_inj)

    def reset(self):
        self.t = 0.0
        self.i_inj = []
        self.current = 0.0
        self.v = -65.0 * np.ones(self.n_x)
        self.n = self.init_n * np.ones(self.n_x)
        self.m = self.init_m * np.ones(self.n_x)
        self.h = self.init_h * np.ones(self.n_x)


class CurrentInjector(Node):
    def __init__(self, i_fn):
        """
        Constructor for CurrentInjector class.
        :param i_fn: function that returns the current (mA) as a function of time (ms)
        """
        super().__init__()
        self.i_fn = i_fn
        self.i = 0.0

    def step(self, dt=0.01):
        self.t += dt
        self.i = self.i_fn(self.t)

    def reset(self):
        self.t = 0.0
        self.i = 0.0


class Synapse(Node):
    def __init__(self, pre_node, post_node: Neuron, g=10.0, e_syn=0.0, delay=0.0, weight=1.0):
        """
        Constructor for Synapse class.
        :param pre_node: node before the synapse
        :param post_node: node after the synapse
        :param g: conductance (S/cm^2)
        :param e_syn: leak reversal potential (mV)
        :param delay: time delay (ms)
        """
        super().__init__()
        self.pre_node = pre_node
        self.post_node = post_node
        self.g = g
        self.e_syn = e_syn
        self.delay = delay
        self.weight = weight
        self.record = np.zeros(int(self.delay / 0.01) + 1)

    def step(self, dt=0.01):
        self.t += dt
        for i in range(1, len(self.record)):
            self.record[i - 1] = self.record[i]
        if isinstance(self.pre_node, Neuron):
            self.record[-1] = self.weight * self.g * (self.pre_node.v[-1] - self.e_syn)
        elif isinstance(self.pre_node, CurrentInjector):
            self.record[-1] = self.weight * self.pre_node.i
        self.post_node.i_inj.append(self.record[0])

    def reset(self):
        self.t = 0.0
        self.record = np.zeros(int(self.delay / 0.01) + 1)


class Recorder:
    def __init__(self, t, *nodes_recorded: Neuron):
        self.nodes_recorded = nodes_recorded
        self.nodes = set()
        self.dt = 0.01
        self.t = 0
        self.n_t = int(t / self.dt) + 1
        self.v = [np.ones((len(nodes_recorded[i].v), self.n_t)) * nodes_recorded[i].v[0] for i in range(len(nodes_recorded))] if len(nodes_recorded) > 0 else None
        self.i = [np.zeros(self.n_t) for _ in range(len(nodes_recorded))] if len(nodes_recorded) > 0 else None

    def update(self, node: Neuron):
        if len(self.nodes_recorded) <= 0:
            return
        try:
            index = self.nodes_recorded.index(node)
            self.v[index][:, self.t] = node.v
            self.i[index][self.t] = node.current
        except ValueError as e:
            return

    def run_network(self, *synapses):
        """
        run the network for t (ms)
        :param synapses: list of synapses
        :return: None
        """
        for synapse in synapses:
            self.nodes.add(synapse.pre_node)
            self.nodes.add(synapse.post_node)
        for synapse in synapses:
            self.nodes.discard(synapse.pre_node)
        for _ in tqdm(range(1, self.n_t)):
            self.t += 1
            for synapse in synapses:
                synapse.pre_node.step(self.dt)
                self.update(synapse.pre_node)
                synapse.step(self.dt)
            for node in self.nodes:
                node.step(self.dt)
                self.update(node)


if __name__ == '__main__':
    import plotly.graph_objects as go


    def current_fn(t):
        if t < 10:
            return 0.0
        elif t < 20:
            return 20.0
        else:
            return 0.0


    current_injector = CurrentInjector(current_fn)
    neuron = Neuron(l=2000.0)
    synapse = Synapse(current_injector, neuron, delay=5.0)
    recorder = Recorder(100, neuron)
    recorder.run_network(synapse)

    fig = go.Figure(data=[
        go.Scatter(x=np.arange(0, neuron.l, neuron.dx), y=recorder.v[0, :, i], name=f"t={i * recorder.dt:.2f}",
                   mode="lines")
        for i in range(0, recorder.n_t, 200)
    ])
    fig.show()