import math
import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

import numba
from numba import int32, int64
from numba.experimental import jitclass
import networkx as nx

from repast4py import core, space, schedule
from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params
from repast4py.network import write_network, read_network
from repast4py.space import DiscretePoint
from repast4py.space import BorderType, OccupancyType
from repast4py.random import default_rng as rng


def generate_network_file(fname: str, n_ranks: int, n_agents: int):
    g = nx.connected_watts_strogatz_graph(n_agents, 2, 0.25)
    try:
        import nxmetis
        write_network(g, 'rumor_network', fname, 1, partition_method='metis')
    except ImportError:
        write_network(g, 'rumor_network', fname, 1)


class Neuron(core.Agent):
    TYPE = 0

    def __init__(self, nid: int, agent_type: int, rank: int, received_misfolding=False):
        super().__init__(nid, agent_type, rank)
        self.received_misfolding = received_misfolding
        self.state = 0
        self.counter = 0
        self.alpha_synuclein_level = 200
        self.misfolding_level = 0
        self.oligomer_level = 0
        self.lewy_bodies_level = 0

    def save(self):
        return (self.uid, self.received_misfolding,
                self.alpha_synuclein_level, self.misfolding_level, self.oligomer_level, self.lewy_bodies_level)

    def update(self, neuron, alphasyn, misf, olig, lewy):
        """Updates the state of this agent when it is a ghost
        agent on some rank other than its local one.
        """

        if not self.received_misfolding and neuron:
            # Only update if the received neuron state has changed from false to true
            if not model.contains(self):
                model.neuron_spreaders.append(self)
            self.received_misfolding = neuron
            self.alpha_synuclein_level = alphasyn
            self.misfolding_level = misf
            self.oligomer_level = olig
            self.lewy_bodies_level = lewy

    def step(self):
        random = rng.uniform()
        # Alpha-syn is increased when there are no Lewy bodies yet
        if self.lewy_bodies_level < 50:
            self.alpha_synuclein_level += rng.integers(1, 500)
        # Chance of alpha-syn reduction
        if random < 0.5 and self.alpha_synuclein_level > 1000:
            self.alpha_synuclein_level -= rng.integers(1, 1000)
        # The alpha-syn misfolds with a lower probability, which increases with its concentration
        # Misfolding increases the misfolding level and decreases normal alpha-syn
        if self.alpha_synuclein_level > 1000 and random + self.alpha_synuclein_level / 100000:
            add_remove = rng.integers(1, 250)
            self.misfolding_level += add_remove
            self.alpha_synuclein_level -= add_remove

        # Formation of alpha-syn oligomers, probability is increased based on the levels
        if (self.misfolding_level > 1000 and
                rng.uniform() < 0.05 + self.misfolding_level / 1000 + self.oligomer_level / 1000):
            variation = rng.integers(0, 50)
            self.misfolding_level -= variation * 10
            self.oligomer_level += variation

        # Formation of Lewy bodies, probability is increased based on the levels
        if (self.oligomer_level > 500 and
                rng.uniform() < 0.02 + self.oligomer_level / 1000 + self.lewy_bodies_level / 1000):
            variation = rng.integers(0, 5)
            self.oligomer_level -= variation * 10
            self.lewy_bodies_level += variation

        # Neuron internal mechanisms try to reduce the amount of bad proteins
        if self.misfolding_level > 100:
            self.misfolding_level -= rng.integers(50, 100)
        if self.oligomer_level > 5:
            self.oligomer_level -= rng.integers(1, 5)

        if self.misfolding_level < 500 and model.contains(self):
            model.neuron_spreaders.remove(self)
            model.counts.new_neuron_spreaders -= 1
            self.received_misfolding = False
            return -1
        if self.misfolding_level > 500 and not model.contains(self):
            model.neuron_spreaders.append(self)
            model.counts.new_neuron_spreaders += 1
            return 1
        return 0


def create_neuron_agent(nid, agent_type, rank, **kwargs):
    return Neuron(nid, agent_type, rank)


@dataclass
class NeuronCounts:
    total_neuron_spreaders: int
    new_neuron_spreaders: int


@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]


spec = [
    ('mo', int32[:]),
    ('no', int32[:]),
    ('xmin', int32),
    ('ymin', int32),
    ('ymax', int32),
    ('xmax', int32)
]


@jitclass(spec)
class GridNghFinder:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.mo = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def find(self, x, y):
        xs = self.mo + x
        ys = self.no + y

        xd = (xs >= self.xmin) & (xs <= self.xmax)
        xs = xs[xd]
        ys = ys[xd]

        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)


class Alpha(core.Agent):
    TYPE = 1
    OFFSETS = 2
    RNDOFFSETS = np.array([-1, 1])

    def __init__(self, a_id: int, rank: int, pt: DiscretePoint, alpha_level: int, misfoldin_level: int):
        super().__init__(id=a_id, type=Alpha.TYPE, rank=rank)
        self.alpha_synuclein_level = alpha_level
        self.misfolding_level = misfoldin_level
        self.pt = pt
        self.fusion = False

    def step(self):
        grid = model.grid
        xy_dirs = rng.choice(Alpha.RNDOFFSETS, size=2)
        # model.move(self, self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1])
        self.pt = grid.get_location(self)
        nghs = model.ngh_finder.find(self.pt.x, self.pt.y)
        at = DiscretePoint(0, 0)
        for ngh in nghs:
            at._reset_from_array(ngh)
            for obj in grid.get_agents(at):
                if self != obj and obj.uid[1] == Neuron.TYPE:
                    obj.misfolding_level += self.misfolding_level
                    obj.alpha_synuclein_level += self.alpha_synuclein_level
                    # print("neurone",obj)
                    self.fusion = True

    def save(self) -> Tuple:
        return self.uid, self.fusion, self.pt.coordinates, self.alpha_synuclein_level, self.misfolding_level


agent_cache = {}


def restore_agent(agent_data: Tuple):
    uid = agent_data[0]

    if uid[1] == Alpha.TYPE:
        pt_array = agent_data[2]
        pt = DiscretePoint(pt_array[0], pt_array[1], 0)
        alpha_level = agent_data[3]
        mis = agent_data[4]
        if uid in agent_cache:
            alpha = agent_cache[uid]
        else:
            alpha = Alpha(uid[0], uid[2], pt, alpha_level, mis)
            agent_cache[uid] = alpha

        alpha.fusion = agent_data[1]
        alpha.pt = pt
        return alpha
    if uid[1] == Neuron.TYPE:
        neuron = Neuron(uid[0], uid[1], uid[2], agent_data[1])
        neuron.alpha_synuclein_level = agent_data[3]
        neuron.misfolding_level = agent_data[3]
        neuron.oligomer_level = agent_data[4]
        neuron.lewy_bodies_level = agent_data[5]
        return neuron


@dataclass
class Counts:
    alpha: int = 0
    alphaMis: int = 0
    oligomer: int = 0


@dataclass
class NeuronCounts:
    total_neuron_spreaders: int
    new_neuron_spreaders: int


class Model:

    def __init__(self, comm, params):
        self.params = params

        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()
        self.protein_counter = params['alpha.count']
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        fpath = params['file_rete']
        self.context = ctx.SharedContext(comm)
        read_network(fpath, self.context, create_neuron_agent, restore_agent)
        self.net = self.context.get_projection('neuron_network')

        self.neuron_spreaders = []

        neuroned_count = len(self.neuron_spreaders)
        self.counts = NeuronCounts(neuroned_count, neuroned_count)
        # loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        # self.data_set = logging.ReducingDataSet(loggers, comm, params['counts_file'])
        # self.data_set.log(0)
        self.neuron_prob = params['neuron_probability']

        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=1, comm=comm)
        self.context.add_projection(self.grid)

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        world_size = comm.Get_size()

        total_protein_count = int(params['alpha.count'] / world_size)

        local_bounds = self.grid.get_local_bounds()
        random = rng.integers(1000, 5000)
        for i in range(total_protein_count):
            x = int(rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent))
            y = int(rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent))
            pt = DiscretePoint(x, y, 0)
            a = Alpha(i, self.rank, pt, random, int(random / 100))
            self.context.add(a)
            self.grid.move(a, pt)
        # for a in self.context.agents():
        #     x = int(rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent))
        #     y = int(rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent))
        #     pt = DiscretePoint(x, y, 0)
        #     self.grid.move(a, pt)

    def get_neighbors(self, agent):
        temp = []
        for ne in self.net.graph.nodes:
            if ne.uid[0] == agent.uid[0] and ne.uid[1] == agent.uid[1] and ne.uid[2] == agent.uid[2]:
                for ngh in self.net.graph.neighbors(ne):
                    temp.append(ngh)
        return temp

    def get_grid_agents(self):
        temp = []
        height = self.params['world.height']
        width = self.params['world.width']
        for x in range(height):
            for y in range(width):
                agent = self.grid.get_agent(DiscretePoint(x, y, 0))
                if agent and agent not in temp:
                    temp.append(agent)
        return temp

    def search_agent(self, id, type):
        if self.rank == 0:
            if self.context.agent((id, type, 0)):
                return self.context.agent((id, type, 0))
            elif self.context.agent((id, type, 1)):
                return self.context.agent((id, type, 1))
            elif self.context.agent((id, type, 2)):
                return self.context.agent((id, type, 2))
            elif self.context.agent((id, type, 3)):
                return self.context.agent((id, type, 3))

    def step(self):
        self.neuron_step()
        fusion_cell = []
        # faccio lo step degli agenti alpha e durante lo step potrebbero incontrare un agente neurone
        for a in self.context.agents(Alpha.TYPE):
            a.step()
            if a.fusion:
                fusion_cell.append(a)

        # rimuovo le alpha con stato fusion ovvero che hanno incontrato un neurone
        for a in fusion_cell:
            self.context.remove(a)
        if self.runner.schedule.tick % 10 == 0:
            local_bounds = self.grid.get_local_bounds()
            for _ in range(5):
                self.protein_counter += 1
                x = int(rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent))
                y = int(rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent))
                pt = DiscretePoint(x, y, 0)
                a = Alpha(self.protein_counter, self.rank, pt, 2000, 50)
                self.context.add(a)
                self.grid.move(a, pt)
        self.context.synchronize(restore_agent)

    def neuron_step(self):
        new_neuron_spreaders = []
        for agent in self.neuron_spreaders:
            if agent.misfolding_level > 500:
                for ngh in self.net.graph.neighbors(agent):
                    # only update agents local to this rank
                    if not ngh.received_misfolding and ngh.local_rank == self.rank:
                        variation = rng.integers(0, 200)
                        ngh.misfolding_level += variation
                        agent.misfolding_level -= variation
                        # ngh.received_misfolding
                        if not self.contains(ngh):
                            new_neuron_spreaders.append(ngh)

        agent_removed = 0
        for agent in self.context.agents(Neuron.TYPE):
            agent_removed += agent.step()
        self.neuron_spreaders += new_neuron_spreaders
        self.counts.new_neuron_spreaders = len(new_neuron_spreaders)
        self.counts.total_neuron_spreaders += self.counts.new_neuron_spreaders + agent_removed
        # self.data_set.log(self.runner.schedule.tick)

    def contains(self, a):
        for agent in self.neuron_spreaders:
            if agent.uid[0] == a.uid[0] and agent.uid[2] == a.uid[2]:
                return True
        return False

    def log_counts(self, tick):
        # Get the current number of zombies and humans and log
        self.alpha_protein = 0
        self.alpha_mis_protein = 0
        self.alpha_oligomer = 0
        self.counts.oligomer = 0
        self.counts.alpha = 0
        # dead_cells = []
        self.counts.alphaMis = 0

        for a in self.context.agents(Alpha.TYPE):
            if a.energy < 1:
                # dead_cells.append(a)
                self.context.remove(a)
            else:
                if a.state == 0:
                    self.counts.alpha += 1
                    self.alpha_protein += 1
                if a.state == 1:
                    self.counts.alphaMis += 1
                    self.alpha_mis_protein += 1
                if a.state == 2:
                    self.alpha_oligomer += 1
                    self.counts.oligomer += 1
        self.data_set.log(tick)

        if self.rank == 0:
            # neuron = self.search_agent(0, 1)
            self.alpha_logger.log_row(tick, self.alpha_protein, self.alpha_mis_protein, self.alpha_oligomer)

            agent = self.search_agent(0, 0)
            if agent:
                self.alpha_position.log_row(tick, agent.uid, self.grid.get_location(agent).x,
                                            self.grid.get_location(agent).y, agent.state)

        # for a in dead_cells:
        # self.context.remove(a)
        folding = np.zeros(1, dtype='int64')
        misfolding = np.zeros(1, dtype='int64')
        oligomers = np.zeros(1, dtype='int64')
        self.comm.Reduce(np.array([self.alpha_protein], dtype='int64'), folding, op=MPI.SUM, root=0)
        self.comm.Reduce(np.array([self.alpha_mis_protein], dtype='int64'), misfolding, op=MPI.SUM, root=0)
        self.comm.Reduce(np.array([self.alpha_mis_protein], dtype='int64'), oligomers, op=MPI.SUM, root=0)

    def move(self, agent, x, y):
        agent.pt = DiscretePoint(x, y, 0)
        self.grid.move(agent, DiscretePoint(int(math.floor(x)), int(math.floor(y))))

    def at_end(self):
        # self.data_set.close()
        for agent in self.context.agents(Neuron.TYPE):
            print(agent, agent.alpha_synuclein_level, agent.misfolding_level, agent.oligomer_level,
                  agent.lewyBodies_level)
        print(self.context.size([Alpha.TYPE, Neuron.TYPE]))
        # for agent in self.context.agents(Alpha.TYPE):
        #     print(agent,agent.misfolding_level,agent.alpha_synuclein_level)

    def start(self):
        self.runner.execute()


model: Model


def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    # generate_network_file("rete",4,20)
    model.start()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)
