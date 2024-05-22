import math
import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass


import numba
from numba import int32, int64
from numba.experimental import jitclass
import networkx as nx

from repast4py import core, space, schedule,logging
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
        self.alpha_synuclein_level = 0
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
        if self.lewy_bodies_level>50:
            self.alpha_synuclein_level+=500
            self.misfolding_level+=250
        # Chance of alpha-syn reduction
        if random < 0.5 and self.alpha_synuclein_level > 1000:
            self.alpha_synuclein_level -= rng.integers(1, 1000)
        # The alpha-syn misfolds with a lower probability, which increases with its concentration
        # Misfolding increases the misfolding level and decreases normal alpha-syn
        if self.alpha_synuclein_level > 1000 and random < 0.3 + self.alpha_synuclein_level / 100000 + self.oligomer_level / 1000 + self.lewy_bodies_level/100 + params['misfolding_pb']/1000:
            add_remove = rng.integers(1, 250)
            self.misfolding_level += add_remove
            self.alpha_synuclein_level -= add_remove

        # Formation of alpha-syn oligomers, probability is increased based on the levels
        if (self.misfolding_level > 1000 and
                rng.uniform() < 0.05 + self.misfolding_level / 1000 + self.oligomer_level / 1000 + self.lewy_bodies_level/100 + params['oligomers_pb']/1000):
            variation = rng.integers(0, 50)
            self.misfolding_level -= variation * 10
            self.oligomer_level += variation

        # Formation of Lewy bodies, probability is increased based on the levels
        if (self.oligomer_level > 300 and
                rng.uniform() < 0.02 + self.oligomer_level / 1000 + self.lewy_bodies_level / 1000 +params['lewy_bodies_pb']/1000):
            variation = rng.integers(0, 5)
            self.oligomer_level -= variation * 10
            self.lewy_bodies_level += variation

        # Neuron internal mechanisms try to reduce the amount of bad proteins
        if self.misfolding_level > 100:
            self.misfolding_level -= rng.integers(50, 100)
        if self.oligomer_level > 10:
            self.oligomer_level -= rng.integers(1, 5)

        if self.misfolding_level < 500 and model.contains(self):
            model.neuron_spreaders.remove(self)
            # model.counts.new_neuron_spreaders -= 1
            self.received_misfolding = False
            return -1
        if self.misfolding_level > 500 and not model.contains(self):
            model.neuron_spreaders.append(self)
            # model.counts.new_neuron_spreaders += 1
            return 1
        if self.misfolding_level>500:
            self.state=1
            if self.oligomer_level>150:
                self.state=2
                if self.lewy_bodies_level>50:
                    self.state=3
        return 0


def create_neuron_agent(nid, agent_type, rank, **kwargs):
    return Neuron(nid, agent_type, rank)


@dataclass
class AgentCount:
    alpha_count:int=0
    bifido_count:int=0
    alpha_gut_count:int=0
    LPS_count:int=0
    gram_negative_count:int=0

@dataclass 
class NeuronStateCount:
    balanced_state:int=0
    misfolding_state:int=0
    oligomer_state:int=0
    lewy_bodies_state:int=0


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
    """Represents a set of Alpha-synuclein molecules"""

    TYPE = 1
    OFFSETS = 2
    RNDOFFSETS = np.array([-1, 1])

    def __init__(self, a_id: int, rank: int, pt: DiscretePoint, alpha_level: int, misfolding_level: int):
        super().__init__(id=a_id, type=Alpha.TYPE, rank=rank)

        self.alpha_synuclein_level = alpha_level
        self.misfolding_level = misfolding_level
        self.pt = pt
        self.fusion = False
        self.spread = False
        self.energy = 1000
        self.is_alive = True
        self.in_CNS = True
        self.came_from_gut=False

    def step(self):
        grid = model.grid if self.in_CNS else model.gut_grid

        xy_dirs = rng.choice(Alpha.RNDOFFSETS, size=2)
        if self.in_CNS:
            model.move(self, self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1])
        else:
            model.move_gut(self, self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1])
        self.pt = grid.get_location(self)
        nghs = model.ngh_finder.find(self.pt.x, self.pt.y)
        at = DiscretePoint(0, 0)
        for ngh in nghs:
            at._reset_from_array(ngh)
            for agent in grid.get_agents(at):
                if self != agent and agent.uid[1] == Neuron.TYPE:
                    agent.misfolding_level += self.misfolding_level
                    agent.alpha_synuclein_level += self.alpha_synuclein_level
                    # print("neurone",obj)
                    self.fusion = True

        if self.energy<0 :
            self.is_alive = False
        if model.runner.schedule.tick % params['tick_life_alpha']:
            if self.energy > 0:
                self.energy-=rng.uniform()

    def save(self) -> Tuple:
        return self.uid, self.fusion, self.pt.coordinates, self.alpha_synuclein_level, self.misfolding_level, self.energy


class GramNegative(core.Agent):
    TYPE = 2
    RNDOFFSETS = np.array([-1, 1])

    def __init__(self, a_id: int, rank: int, pt: DiscretePoint):
        super().__init__(id=a_id, type=GramNegative.TYPE, rank=rank)
        self.pt = pt
        self.generateLPS = False
        self.remove = False
        self.release = False

    def step(self):
        xy_dirs = rng.choice(GramNegative.RNDOFFSETS, size=2)
        model.move_gut(self, self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1])
        self.pt = model.gut_grid.get_location(self)
        if rng.uniform() < 0.06 + model.lps_count / 100000 + params['dysboisi_probability'] / 100+ model.runner.schedule.tick / 10000:
            self.generateLPS = True

        death_pb = rng.uniform()
        if death_pb < 0.01:
            self.remove = True
            if death_pb < 0.0001 + params['release_probability'] / 5000:
                self.release = True

    def save(self) -> Tuple:
        return self.uid, self.pt.coordinates


class LPS(core.Agent):
    TYPE = 3
    RNDOFFSETS = np.array([-1, 1])

    def __init__(self, a_id: int, rank: int, pt: DiscretePoint):
        super().__init__(id=a_id, type=LPS.TYPE, rank=rank)
        self.pt = pt
        self.remove = False

    def step(self):
        xy_dirs = rng.choice(LPS.RNDOFFSETS, size=2)
        model.move_gut(self, self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1])
        self.pt = model.gut_grid.get_location(self)

    def save(self) -> Tuple:
        return self.uid, self.pt.coordinates


class Bifidobacteria(core.Agent):
    TYPE = 4
    RNDOFFSETS = np.array([-1, 1])

    def __init__(self, a_id: int, rank: int, pt: DiscretePoint):
        super().__init__(id=a_id, type=Bifidobacteria.TYPE, rank=rank)
        self.pt = pt
        self.remove = False

    def step(self):
        grid = model.gut_grid
        xy_dirs = rng.choice(LPS.RNDOFFSETS, size=2)
        model.move_gut(self, self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1])
        self.pt = grid.get_location(self)
        nghs = model.ngh_finder.find(self.pt.x, self.pt.y)
        at = DiscretePoint(0, 0, 0)
        for ngh in nghs:
            at._reset_from_array(ngh)
            for obj in grid.get_agents(at):
                if self != obj and (obj.uid[1] == LPS.TYPE or obj.uid[1] == GramNegative.TYPE):
                    obj.remove = True
        if rng.uniform() < params['pr_dead_bifido']:
            self.remove = True

    def save(self) -> Tuple:
        return self.uid, self.pt.coordinates


class GutNeuron(core.Agent):
    TYPE = 5

    def __init__(self, nid: int, rank: int, pt: DiscretePoint):
        super().__init__(id=nid, type=GutNeuron.TYPE, rank=rank)
        self.pt = pt

    def step(self):
        nghs = model.ngh_finder.find(self.pt.x, self.pt.y)
        at = DiscretePoint(0, 0, 0)
        for ngh in nghs:
            at._reset_from_array(ngh)
            for obj in model.gut_grid.get_agents(at):
                if self != obj and obj.uid[1] == Alpha.TYPE:
                    obj.spread = True

    def save(self) -> Tuple:
        return self.uid, self.pt.coordinates


cns_agent_cache = {}


def restore_agent(agent_data: Tuple):
    uid = agent_data[0]

    if uid[1] == Alpha.TYPE:
        pt_array = agent_data[2]
        pt = DiscretePoint(pt_array[0], pt_array[1], 0)
        alpha_level = agent_data[3]
        mis = agent_data[4]
        if uid in cns_agent_cache:
            alpha = cns_agent_cache[uid]
        else:
            alpha = Alpha(uid[0], uid[2], pt, alpha_level, mis)
            cns_agent_cache[uid] = alpha
        alpha.energy = agent_data[5]
        alpha.fusion = agent_data[1]
        alpha.pt = pt
        alpha.in_CNS = True
        return alpha

    if uid[1] == Neuron.TYPE:
        neuron = Neuron(uid[0], uid[1], uid[2], agent_data[1])
        neuron.alpha_synuclein_level = agent_data[3]
        neuron.misfolding_level = agent_data[3]
        neuron.oligomer_level = agent_data[4]
        neuron.lewy_bodies_level = agent_data[5]
        return neuron


gut_agent_cache = {}


def restore_agent_gut(agent_data: Tuple):
    uid = agent_data[0]

    if uid[1] == GramNegative.TYPE:
        pt_array = agent_data[1]
        pt = DiscretePoint(pt_array[0], pt_array[1], 0)
        if uid in gut_agent_cache:
            cell = gut_agent_cache[uid]
        else:
            cell = GramNegative(uid[0], uid[2], pt)
            gut_agent_cache[uid] = cell
        cell.pt = pt
        return cell

    if uid[1] == LPS.TYPE:
        pt_array = agent_data[1]
        pt = DiscretePoint(pt_array[0], pt_array[1], 0)
        if uid in gut_agent_cache:
            lps = gut_agent_cache[uid]
        else:
            lps = LPS(uid[0], uid[2], pt)
            gut_agent_cache[uid] = lps
        lps.pt = pt
        return lps

    if uid[1] == Bifidobacteria.TYPE:
        pt_array = agent_data[1]
        pt = DiscretePoint(pt_array[0], pt_array[1], 0)
        if uid in gut_agent_cache:
            bifido = gut_agent_cache[uid]
        else:
            bifido = Bifidobacteria(uid[0], uid[2], pt)
            gut_agent_cache[uid] = bifido
        bifido.pt = pt
        return bifido

    if uid[1] == Alpha.TYPE:
        pt_array = agent_data[2]
        pt = DiscretePoint(pt_array[0], pt_array[1], 0)
        alpha_level = agent_data[3]
        mis = agent_data[4]
        if uid in gut_agent_cache:
            alpha = gut_agent_cache[uid]
        else:
            alpha = Alpha(uid[0], uid[2], pt, alpha_level, mis)
            gut_agent_cache[uid] = alpha
        alpha.energy = agent_data[5]
        alpha.fusion = agent_data[1]
        alpha.pt = pt
        alpha.in_CNS = False
        return alpha

    if uid[1] == GutNeuron.TYPE:
        pt_array = agent_data[1]
        pt = DiscretePoint(pt_array[0], pt_array[1], 0)
        if uid in gut_agent_cache:
            neuron = gut_agent_cache[uid]
        else:
            neuron = GutNeuron(uid[0], uid[2], pt)
            gut_agent_cache[uid] = neuron
        neuron.pt = pt
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

    def __init__(self, comm: MPI.Intracomm, params):
        self.params = params

        self.comm = comm
        self.rank = comm.Get_rank()
        self.protein_counter = params['alpha.count']

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        self.neuron_spreaders = []
       
        
        


        # ================ CNS CONTEXT ================
        self.context = ctx.SharedContext(comm)
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=1, comm=comm)
        self.context.add_projection(self.grid)

        read_network(params['file_cns_network'], self.context, create_neuron_agent, restore_agent)
        self.net = self.context.get_projection('neuron_network')

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        self.alpha_protein = 0
        self.alpha_mis_protein = 0
        self.alpha_oligomer = 0

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

        for agent in self.context.agents(Neuron.TYPE):
            x = int(rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent))
            y = int(rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent))
            pt = DiscretePoint(x, y, 0)
            self.grid.move(agent, pt)

        # ================ GUT CONTEXT ================
        self.gut_context = ctx.SharedContext(comm)
        gut_box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.gut_grid = space.SharedGrid('gut_grid', bounds=gut_box, borders=BorderType.Sticky,
                                         occupancy=OccupancyType.Multiple,
                                         buffer_size=1, comm=comm)
        self.gut_context.add_projection(self.gut_grid)

        gut_local_bounds = self.gut_grid.get_local_bounds()
        self.lps_count = 0
       
        self.passedProteins=0
        # Generate Gut neurons
        for i in range(params['gut_neuron']):
            x = int(rng.uniform(gut_local_bounds.xmin, gut_local_bounds.xmin + gut_local_bounds.xextent))
            y = int(rng.uniform(gut_local_bounds.ymin, gut_local_bounds.ymin + gut_local_bounds.yextent))
            pt = DiscretePoint(x, y, 0)

            gut_neuron = GutNeuron(i, self.rank, pt)
            self.gut_context.add(gut_neuron)
            self.gut_grid.move(gut_neuron, pt)

        # Generate Gram-negative
        for i in range(params['gram_negative']):
            x = int(rng.uniform(gut_local_bounds.xmin, gut_local_bounds.xmin + gut_local_bounds.xextent))
            y = int(rng.uniform(gut_local_bounds.ymin, gut_local_bounds.ymin + gut_local_bounds.yextent))
            pt = DiscretePoint(x, y, 0)

            g_neg = GramNegative(i, self.rank, pt)
            self.gut_context.add(g_neg)
            self.gut_grid.move(g_neg, pt)

        # Generate Bifidobacteria
        for i in range(params['bifidobacteria']):
            x = int(rng.uniform(gut_local_bounds.xmin, gut_local_bounds.xmin + gut_local_bounds.xextent))
            y = int(rng.uniform(gut_local_bounds.ymin, gut_local_bounds.ymin + gut_local_bounds.yextent))
            pt = DiscretePoint(x, y, 0)

            bifido = Bifidobacteria(i, self.rank, pt)
            self.gut_context.add(bifido)
            self.gut_grid.move(bifido, pt)

        # Generate Alpha
        for _ in range(params['alpha_gut']):
            x = int(rng.uniform(gut_local_bounds.xmin, gut_local_bounds.xmin + gut_local_bounds.xextent))
            y = int(rng.uniform(gut_local_bounds.ymin, gut_local_bounds.ymin + gut_local_bounds.yextent))
            pt = DiscretePoint(x, y, 0)
            self.protein_counter += 1
            alpha = Alpha(self.protein_counter, self.rank, pt, params['starting_folding_level'], params['starting_oligomer_level'])
            alpha.in_CNS = False
            self.gut_context.add(alpha)
            self.gut_grid.move(alpha, pt)

        self.gram_count = self.gut_context.size([GramNegative.TYPE])[2]
        self.bifido_count = self.gut_context.size([Bifidobacteria.TYPE])[4]
        self.gut_alpha_count = 0
        
        self.counts = AgentCount(self.context.size([Alpha.TYPE])[1],params['bifidobacteria'],0,0,params['gram_negative'])
        loggers=logging.create_loggers(self.counts,op=MPI.SUM, rank=self.rank)
        self.data_set=logging.ReducingDataSet(loggers,comm,params['counts_file'])
        self.data_set.log(0)

        self.neuron_counter= logging.TabularLogger(comm,params['nueron_state'],['tick','balanced','misfolding','oligomer','lewy_bodies'])
        # initialize the logging
        self.agent_gut_pos = logging.TabularLogger(comm, params['agent_pos_gut'], ['tick', 'agent_id', 'x', 'y'])
        self.agent_cns_pos = logging.TabularLogger(comm, params['agent_pos_cns'], ['tick', 'agent_id', 'x', 'y','neuron_state','came_from_gut'])

    def get_neighbors(self, agent):
        temp = []
        for ne in self.net.graph.nodes:
            if ne.uid[0] == agent.uid[0] and ne.uid[1] == agent.uid[1] and ne.uid[2] == agent.uid[2]:
                for ngh in self.net.graph.neighbors(ne):
                    temp.append(ngh)
        return temp

    def step(self):
        if params['attiva_CNS']:
            self.neuron_step()

            self.log_nueron()
            fusion_cell = []
            remove_cell=[]
        # Alpha agents step, they are added to the list if they encountered a neuron
            for a in self.context.agents(Alpha.TYPE):
                bad_alpha = a
                a.step()
                if a.fusion or not a.is_alive:
                    fusion_cell.append(a)
               

        # Remove Alpha agents that encountered a neuron
            for a in fusion_cell:
                self.context.remove(a)
         
        #Generate Alpha in CNS 
            if self.runner.schedule.tick % params['alpha_generation_CNS'] == 0:
                local_bounds = self.grid.get_local_bounds()
                for _ in range(params['alpha_cns']):
                    self.protein_counter += 1
                    x = int(rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent))
                    y = int(rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent))
                    pt = DiscretePoint(x, y, 0)
                    a = Alpha(self.protein_counter, self.rank, pt, params['folding_level_generation_alpha'], params['misfolding_level_generation_alpha'])
                    self.context.add(a)
                    self.grid.move(a, pt)
        if params['attiva_gut']:
            self.gut_step()
        self.log()
    
    def log(self):
        alpha_gut_count=0
        LPS_count=0
        if self.gut_context.contains_type(LPS.TYPE):
            LPS_count=self.gut_context.size([LPS.TYPE])[LPS.TYPE]
        if(self.gut_context.contains_type(Alpha.TYPE)):
            alpha_gut_count=self.gut_context.size([Alpha.TYPE])[1]
        self.counts.bifido_count=self.gut_context.size([Bifidobacteria.TYPE])[Bifidobacteria.TYPE]
        self.counts.alpha_count=self.context.size([Alpha.TYPE])[1]
        self.counts.alpha_gut_count=alpha_gut_count
        self.counts.gram_negative_count=self.gut_context.size([GramNegative.TYPE])[GramNegative.TYPE]
        self.counts.LPS_count=LPS_count
        self.data_set.log(self.runner.schedule.tick)
        for agent in self.gut_context.agents(GramNegative.TYPE):
            self.agent_gut_pos.log_row(self.runner.schedule.tick,agent.TYPE,self.gut_grid.get_location(agent).x,self.gut_grid.get_location(agent).y)
        for agent in self.gut_context.agents(Bifidobacteria.TYPE):
            self.agent_gut_pos.log_row(self.runner.schedule.tick,agent.TYPE,self.gut_grid.get_location(agent).x,self.gut_grid.get_location(agent).y)
        if self.gut_context.contains_type(LPS.TYPE):
            for agent in self.gut_context.agents(LPS.TYPE):
                self.agent_gut_pos.log_row(self.runner.schedule.tick,agent.TYPE,self.gut_grid.get_location(agent).x,self.gut_grid.get_location(agent).y)
        for agent in self.gut_context.agents(GutNeuron.TYPE):
            self.agent_gut_pos.log_row(self.runner.schedule.tick,agent.TYPE,self.gut_grid.get_location(agent).x,self.gut_grid.get_location(agent).y)
        for agent in self.gut_context.agents(Alpha.TYPE):
            self.agent_gut_pos.log_row(self.runner.schedule.tick,agent.TYPE,self.gut_grid.get_location(agent).x,self.gut_grid.get_location(agent).y)
        self.agent_gut_pos.write()
        for agent in self.context.agents(Neuron.TYPE):
            self.agent_cns_pos.log_row(self.runner.schedule.tick,agent.TYPE,self.grid.get_location(agent).x,self.grid.get_location(agent).y,agent.state,False)
        for agent in self.context.agents(Alpha.TYPE):
            self.agent_cns_pos.log_row(self.runner.schedule.tick,agent.TYPE,self.grid.get_location(agent).x,self.grid.get_location(agent).y,0,agent.came_from_gut)
        self.agent_cns_pos.write()
        
    def log_nueron(self):
        balanced=0
        misfolding=0
        oligomer=0
        lewy_bodies=0
        self.neuron_counter
        for agent in self.context.agents(Neuron.TYPE):
            if agent.state==0:
                balanced+=1
            if agent.state==1:
                misfolding+=1
            if agent.state==2:
                oligomer+=1
            if agent.state==3:
                lewy_bodies+=1
        folding = np.zeros(1, dtype='int64')
        misfo = np.zeros(1, dtype='int64')
        oligo = np.zeros(1, dtype='int64')
        lewy= np.zeros(1,dtype='int64')
        self.comm.Reduce(np.array([balanced], dtype='int64'), folding, op=MPI.SUM, root=0)
        self.comm.Reduce(np.array([misfolding], dtype='int64'), misfo, op=MPI.SUM, root=0)
        self.comm.Reduce(np.array([oligomer], dtype='int64'), oligo, op=MPI.SUM, root=0)
        self.comm.Reduce(np.array([lewy_bodies],dtype='int64'),lewy, op=MPI.SUM, root=0)
        if self.rank==0:
            self.neuron_counter.log_row(self.runner.schedule.tick,folding[0],misfo[0],oligo[0],lewy[0])
        self.neuron_counter.write()

    def neuron_step(self):
        new_neuron_spreaders = []
        for agent in self.neuron_spreaders:
            if agent.misfolding_level > 500:
                for ngh in self.net.graph.neighbors(agent):
                    # Only update agents local to this rank
                    if not ngh.received_misfolding and ngh.local_rank == self.rank:
                        variation = rng.integers(0, 200)
                        ngh.misfolding_level += variation
                        agent.misfolding_level -= variation
                        if not self.contains(ngh):
                            new_neuron_spreaders.append(ngh)

        for agent in self.context.agents(Neuron.TYPE):
            agent.step()
        

    def gut_step(self):
        if self.runner.schedule.tick % 10 == 0:
            self.generate_gut_agents()

        remove_agents = []
        spread_alpha = []
        if self.gut_context.contains_type(Alpha.TYPE):
            for alpha in self.gut_context.agents(Alpha.TYPE):
                if alpha.is_alive:
                    alpha.step()
                    if alpha.spread:
                        spread_alpha.append(alpha)
                else:
                    remove_agents.append(alpha)

        for gut_neuron in self.gut_context.agents(GutNeuron.TYPE):
            gut_neuron.step()

        for b in self.gut_context.agents(Bifidobacteria.TYPE):
            if b.remove:
                remove_agents.append(b)
            else:
                b.step()

        if self.gut_context.contains_type(LPS.TYPE):
            for lps in self.gut_context.agents(LPS.TYPE):
                if lps.remove:
                    remove_agents.append(lps)
                else:
                    lps.step()

        for g_neg in self.gut_context.agents(GramNegative.TYPE):
            if g_neg.release:
                self.generate_lps(g_neg)
            if g_neg.remove or g_neg.release:
                remove_agents.append(g_neg)
            else:
                g_neg.step()

        for alpha in remove_agents:
            self.gut_context.remove(alpha)

        for g_neg in self.gut_context.agents(GramNegative.TYPE):
            if g_neg.generateLPS:
                self.lps_count += 1
                lps = LPS(self.lps_count, self.rank, g_neg.pt)
                self.gut_context.add(lps)
                self.gut_grid.move(lps, g_neg.pt)
                g_neg.generateLPS = False

        # Move Alpha agents from gut context to the CNS context
        local_bounds = self.grid.get_local_bounds()
        for alpha in spread_alpha:
            self.gut_context.remove(alpha)
            self.passedProteins+=1
            if params['flag_enabling_move_between_system']:
                x = int(rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent))
                y = int(rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent))
                pt = DiscretePoint(x, y, 0)
                alpha.in_CNS = True
                alpha.came_from_gut=True
                self.context.add(alpha)
                self.grid.move(alpha, pt)

        self.context.synchronize(restore_agent)
        self.gut_context.synchronize(restore_agent_gut)

    def generate_lps(self, g_neg):
        for _ in range(params['lps_release']+int(self.runner.schedule.tick / 100)):
            self.lps_count += 1
            lps = LPS(self.lps_count, self.rank, g_neg.pt)
            self.gut_context.add(lps)
            self.gut_grid.move(lps, g_neg.pt)

    def generate_gut_agents(self):
        local_bounds_gut = self.gut_grid.get_local_bounds()

        # Generate Gram-negative
        for _ in range(params['gram_negative']+ int(self.runner.schedule.tick / 100)):
            x = int(rng.uniform(local_bounds_gut.xmin, local_bounds_gut.xmin + local_bounds_gut.xextent))
            y = int(rng.uniform(local_bounds_gut.ymin, local_bounds_gut.ymin + local_bounds_gut.yextent))
            pt = DiscretePoint(x, y, 0)
            self.gram_count += 1
            cell = GramNegative(self.gram_count, self.rank, pt)
            self.gut_context.add(cell)
            self.gut_grid.move(cell, pt)

        # Generate Bifidobacteria
        for _ in range(params['bifidobacteria']):
            x = int(rng.uniform(local_bounds_gut.xmin, local_bounds_gut.xmin + local_bounds_gut.xextent))
            y = int(rng.uniform(local_bounds_gut.ymin, local_bounds_gut.ymin + local_bounds_gut.yextent))
            pt = DiscretePoint(x, y, 0)
            self.bifido_count += 1
            bifido = Bifidobacteria(self.bifido_count, self.rank, pt)
            self.gut_context.add(bifido)
            self.gut_grid.move(bifido, pt)

        # Generate Alpha
        for _ in range((0 + int(self.lps_count / params['generation_alpha_gut']))):
            x = int(rng.uniform(local_bounds_gut.xmin, local_bounds_gut.xmin + local_bounds_gut.xextent))
            y = int(rng.uniform(local_bounds_gut.ymin, local_bounds_gut.ymin + local_bounds_gut.yextent))
            pt = DiscretePoint(x, y, 0)
            self.protein_counter += 1
            alpha = Alpha(self.protein_counter, self.rank, pt, params['folding_in_gut'], params['oligomers_in_gut'])
            alpha.in_CNS = False
            self.gut_context.add(alpha)
            self.gut_grid.move(alpha, pt)

    def contains(self, a):
        for agent in self.neuron_spreaders:
            if agent.uid[0] == a.uid[0] and agent.uid[2] == a.uid[2]:
                return True
        return False

    def log_counts(self):
        self.counts.oligomer = 0
        self.counts.alpha = 0
        self.counts.alphaMis = 0

        for a in self.context.agents(Alpha.TYPE):
            if a.energy < 1:
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

        folding = np.zeros(1, dtype='int64')
        misfolding = np.zeros(1, dtype='int64')
        oligomers = np.zeros(1, dtype='int64')
        self.comm.Reduce(np.array([self.alpha_protein], dtype='int64'), folding, op=MPI.SUM, root=0)
        self.comm.Reduce(np.array([self.alpha_mis_protein], dtype='int64'), misfolding, op=MPI.SUM, root=0)
        self.comm.Reduce(np.array([self.alpha_mis_protein], dtype='int64'), oligomers, op=MPI.SUM, root=0)

    def move(self, agent, x, y):
        agent.pt = DiscretePoint(x, y, 0)
        self.grid.move(agent, DiscretePoint(int(math.floor(x)), int(math.floor(y))))

    def move_gut(self, agent, x, y):
        agent.pt = DiscretePoint(x, y, 0)
        self.gut_grid.move(agent, DiscretePoint(int(math.floor(x)), int(math.floor(y))))

    def at_end(self):
        self.data_set.close()
        self.agent_gut_pos.close()
        self.agent_cns_pos.close()
        self.neuron_counter.close()
        # self.bifido_pos.close()
        for agent in self.context.agents(Neuron.TYPE):
            print(agent, agent.alpha_synuclein_level, agent.misfolding_level, agent.oligomer_level,
                  agent.lewy_bodies_level)
        # print(self.context.size([Alpha.TYPE, Neuron.TYPE]))
        if params['attiva_gut']:
            print("GUT", self.gut_context.size([LPS.TYPE, GramNegative.TYPE, Bifidobacteria.TYPE, Alpha.TYPE, GutNeuron.TYPE]))
        if params['attiva_CNS']:
            print("CNS", self.context.size([Alpha.TYPE, Neuron.TYPE]))
        print("proteine passate",self.passedProteins)
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
