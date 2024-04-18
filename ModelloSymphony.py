import sys
import math
import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

import numba
from numba import int32, int64
from numba.experimental import jitclass
import networkx as nx
from repast4py import core, space, schedule, logging, random
from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params
import repast4py
from repast4py.network import write_network, read_network
from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType
model = None

def generate_network_file(fname:str, n_ranks:int, n_agents:int):
    g=nx.connected_watts_strogatz_graph(n_agents, 2, 0.25)
    try:
        import nxmetis
        write_network(g, 'rumor_network', fname, 1, partition_method='metis')
    except ImportError:
        write_network(g, 'rumor_network', fname, 1)

class Neuron(core.Agent):
    TYPE=0
    def __init__(self, nid: int, agent_type: int, rank: int, received_misfolding=False):
        super().__init__(nid, agent_type, rank)
        self.received_misfolding = received_misfolding
        self.state=0
        self.counter=0
        self.alpha_synuclein_level=200
        self.misfolding_level=0
        self.oligomer_level=0
        self.lewyBodies_level=0

    def save(self):
        """Saves the state of this agent as tuple.

        A non-ghost agent will save its state using this
        method, and any ghost agents of this agent will
        be updated with that data (self.received_misfolding).

        Returns:
            The agent's state
        """
        return (self.uid, self.received_misfolding,self.alpha_synuclein_level,self.misfolding_level,self.oligomer_level,self.lewyBodies_level)

    def update(self,neuron, alphasyn,misf,olig,lewy):
        """Updates the state of this agent when it is a ghost
        agent on some rank other than its local one.

        Args:
            data: the new agent state (received_misfolding)
        """
       
        if not self.received_misfolding and neuron:
            # only update if the received neuron state
            # has changed from false to true
            if not model.contains(self):
                model.neuron_spreaders.append(self)
            self.received_misfolding = neuron
            self.alpha_synuclein_level=alphasyn
            self.misfolding_level=misf
            self.oligomer_level=olig
            self.lewyBodies_level=lewy
    def step(self): 
        rng = random.default_rng
        # fintanto che non rilevo corpi di levi continuo a creare alphasynucleina in base ad un valore randomico
        if self.lewyBodies_level<50:
            self.alpha_synuclein_level+=random.default_rng.integers(1,500)
        # vengono eliminate le alphasyn
        if rng.uniform() < 0.5 and self.alpha_synuclein_level>1000:
            self.alpha_synuclein_level-=random.default_rng.integers(1,1000)
        # con un probabilità minori rispetto all'eliminazione vado a misfoldare le proteine
        # quando misfoldo aggiungo a misfoling level e riduco le alpha normali
        if rng.uniform() <0.25+self.alpha_synuclein_level/1000000 and self.alpha_synuclein_level>1000:
                add_remove=random.default_rng.integers(1,250)
                self.misfolding_level+=add_remove
                self.alpha_synuclein_level-=add_remove
        
        # da misfolding a oligomero 
        if random.default_rng.uniform()<0.05+self.misfolding_level/1000+self.oligomer_level/1000 and self.misfolding_level>1000:
            add_remove=random.default_rng.integers(0,50)
            self.misfolding_level-=add_remove*10
            self.oligomer_level+=add_remove
        # oligomer -> lewy
        if random.default_rng.uniform() <0.02+self.oligomer_level/1000+self.lewyBodies_level/1000 and self.oligomer_level>500:
            add_remove=random.default_rng.integers(0,5)
            self.oligomer_level-=add_remove*10
            self.lewyBodies_level+=add_remove

        # proteasome 
        proteasome=random.default_rng.uniform()
        if self.misfolding_level>100:
            add_remove=random.default_rng.integers(10,100)
            self.misfolding_level-=add_remove
        if self.oligomer_level>5:
                add_remove=random.default_rng.integers(1,5)
                self.oligomer_level-=add_remove

        if self.misfolding_level<500 and model.contains(self):
            model.neuron_spreaders.remove(self)
            model.countsNeuron.new_neuron_spreaders-=1
            self.received_misfolding=False
            return -1
        if self.misfolding_level>500 and not model.contains(self):
            model.neuron_spreaders.append(self)
            model.countsNeuron.new_neuron_spreaders+=1
            return 1
        return 0

def create_neuron_agent(nid, agent_type, rank, **kwargs):
    return Neuron(nid, agent_type, rank)


def restore_neuron(agent_data):
    uid = agent_data[0]
    neuron=Neuron(uid[0], uid[1], uid[2], agent_data[1])
    neuron.alpha_synuclein_level=agent_data[3]
    neuron.misfolding_level=agent_data[3]
    neuron.oligomer_level=agent_data[4]
    neuron.lewyBodies_level=agent_data[5]
    return neuron



@dataclass
class NeuronCounts:
    total_neuron_spreaders: int
    new_neuron_spreaders: int

@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]
spec=[
    ('mo',int32[:]),
    ('no',int32[:]),
    ('xmin',int32),
    ('ymin',int32),
    ('ymax',int32),
    ('xmax',int32)
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
    TYPE=1
    OFFSETS=2
    RNDOFFSETS = np.array([-1, 1])
    def __init__(self,a_id:int,rank:int,pt:dpt):
        super().__init__(id=a_id,type=Alpha.TYPE,rank=rank)
        self.state=0
        self.energy=20
        self.pt=pt

    def step(self):
        pb=random.default_rng.integers(0,100)
        if self.state==0 and pb<10:
            self.state=1
        if self.state==0:
            self.energy-=1
        if self.state==1:
            self.move()
        
    def updateState(self):
        self.state=2
    def move(self):
        grid=model.grid
        
        # pt=grid.get_location(self)
        # nextPt=self.pt
        # agente=model.grid_agent[random.default_rng.integers(0,len(model.grid_agent))-1]
        # while agente==self:
        #     agente=model.grid_agent[random.default_rng.integers(0,len(model.grid_agent))-1]
        # nextPt=agente.pt

        # dx = 0 if nextPt.x == pt.x else (1 if nextPt.x > pt.x else -1)
        # dy = 0 if nextPt.y == pt.y else (1 if nextPt.y > pt.y else -1)
        # flag=True
        # if not (abs(pt.x-nextPt.x)<=1 and abs(pt.y-nextPt.y)<=1):
        #     model.move(self,pt.x+dx,pt.y+dy)
        #     flag=False
        # if flag:
        #     self.updateState()
        xy_dirs = random.default_rng.choice(Alpha.RNDOFFSETS, size=2)
        model.move(self, self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1])
        self.pt=grid.get_location(self)
        nghs = model.ngh_finder.find(self.pt.x, self.pt.y)
        at = dpt(0, 0)
        count=0
        for ngh in nghs:
            at._reset_from_array(ngh)
            for obj in grid.get_agents(at):
                if self!=obj and obj.uid[1] == Alpha.TYPE:
                    count+=1
        if count>0:
            self.state=2
    def save(self)->Tuple:
        return (self.uid,self.state,self.pt.coordinates)

agent_cache = {}
def restore_agent(agent_data: Tuple):
    uid = agent_data[0]
    
    if uid[1] == Alpha.TYPE:
        pt_array=agent_data[2]
        pt = dpt(pt_array[0], pt_array[1], 0)
        if uid in agent_cache:
            alpha = agent_cache[uid]
        else:
            alpha = Alpha(uid[0], uid[2],pt)
            agent_cache[uid] = alpha

        # restore the agent state from the agent_data tuple
        alpha.state = agent_data[1]
        alpha.pt=pt
        return alpha
    if uid[1]== Neuron.TYPE:
        neuron=Neuron(uid[0], uid[1], uid[2], agent_data[1])
        neuron.alpha_synuclein_level=agent_data[3]
        neuron.misfolding_level=agent_data[3]
        neuron.oligomer_level=agent_data[4]
        neuron.lewyBodies_level=agent_data[5]
    return neuron
@dataclass
class Counts:
    alpha:int=0
    alphaMis:int=0
    oligome:int=0

@dataclass
class NeuronCounts:
    total_neuron_spreaders: int
    new_neuron_spreaders: int

class Model:
    def __init__(self,comm,params):
       
        self.comm=comm
        self.context=ctx.SharedContext(comm)
        self.rank=self.comm.Get_rank()

        self.runner=schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1,1,self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        fpath = params['file_rete']
        self.context = ctx.SharedContext(comm)
        read_network(fpath, self.context, create_neuron_agent, restore_agent)
        self.net = self.context.get_projection('neuron_network')

        self.neuron_spreaders = []
        self.rank = comm.Get_rank()
        # self._seed_neuron(params['initial_neuron_count'], comm)

        neuroned_count = len(self.neuron_spreaders)
        self.countsNeuron = NeuronCounts(neuroned_count, neuroned_count)
        loggers = logging.create_loggers(self.countsNeuron, op=MPI.SUM, rank=self.rank)
        self.data_set_diffousori = logging.ReducingDataSet(loggers, comm, params['counts_diffusori'])
        self.agent_features=logging.TabularLogger(comm,params['agent_features'],['tick','agent','counter'])
        self.data_set_diffousori.log(0)

        self.neuron_prob = params['neuron_probability']

        box=space.BoundingBox(0,params['world.width'],0,params['world.height'],0,0)
        self.grid=space.SharedGrid('grid',bounds=box,borders=BorderType.Sticky,occupancy=OccupancyType.Multiple,buffer_size=1,comm=comm)
        self.context.add_projection(self.grid)

        self.ngh_finder=GridNghFinder(0,0,box.xextent,box.yextent)

        self.counts=Counts()
        loggers=logging.create_loggers(self.counts,op=MPI.SUM,rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['ricerca'])
        
        self.alpha_logger=logging.TabularLogger(comm,params['counts_file'],['tick','folding','misfolding','oligomer'])
        self.alpha_position=logging.TabularLogger(comm,params['counts_pos'],['tick','aget_id','x','y','stato'])
        self.neuron_data=logging.TabularLogger(comm,params['posizioni'],['tick','alpha','misfolding','oligomer','lewy','state','flag'])
        world_size=comm.Get_size()

        total_protein_count=int(params['alpha.count']/world_size)
       
        local_bounds = self.grid.get_local_bounds()
        for i in range(total_protein_count):
            x = int(random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent))
            y = int(random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent))
            pt=dpt(x,y,0)
            a=Alpha(i,self.rank,pt)
            self.context.add(a)
            self.grid.move(a,pt)
        for a in self.context.agents():
                x = int(random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent))
                y = int(random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent))
                pt=dpt(x,y,0)
                self.grid.move(a,pt)
        self.grid_agent=self.getGridAgent()
        # for agent in self.net.graph.nodes:
        #     print (agent, "vicini ")
        #     for ngh in self.net.graph.neighbors(agent):
        #         print(ngh)
    
    def getNeighbours(self,agent):  
        temp=[]
        for ne in self.net.graph.nodes:
            if ne.uid[0]==agent.uid[0] and ne.uid[1]==agent.uid[1] and ne.uid[2]==agent.uid[2]:
                for ngh in self.net.graph.neighbors(ne):
                    temp.append(ngh)
        return temp
    def getGridAgent(self):
        temp=[]
        altezza=params['world.height']
        larghezza=params['world.width']
        for x in range(altezza):
            for y in range(larghezza):
                if self.grid.get_agent(dpt(x,y,0)):
                    if not self.cotains(temp, self.grid.get_agent(dpt(x,y,0))):
                        temp.append(self.grid.get_agent(dpt(x,y,0)))
        return temp
    
    def cotains(self,vettore,elemento):
        for a in vettore:
            if a==elemento:
                return True
        return False

    def searchAgent(self,id,type):
        if(self.rank==0):
            if self.context.agent((id, type, 0)):
                return self.context.agent((id, type, 0))
            elif self.context.agent((id, type, 1)):
                return self.context.agent((id, type, 1))
            elif self.context.agent((id, type, 2)):
                return self.context.agent((id, type, 2))
            elif self.context.agent((id, type, 3)):
                return self.context.agent((id, type, 3))
    def step(self):
        self.nueronStep()
        tick =self.runner.schedule.tick
        self.log_counts(tick)
        self.alpha_logger.write()
        self.neuron_data.write()
        for a in self.context.agents(Alpha.TYPE):
            a.step()
        # rng=repast4py.random.default_rng
        # for i in range(200):
        #     pt=self.grid.get_random_local_pt(rng)
        #     a=Alpha(self.contatore,self.rank,pt)
        #     self.contatore+=1
        #     self.context.add(a)
        #     self.grid.move(a,pt)
        self.context.synchronize(restore_agent)

    def nueronStep(self):
        new_neuron_spreaders = []
        rng = random.default_rng
        for agent in self.neuron_spreaders:
            if agent.misfolding_level>500:
                for ngh in self.net.graph.neighbors(agent):
                    # only update agents local to this rank
                    if not ngh.received_misfolding and ngh.local_rank == self.rank:
                        add_remove=random.default_rng.integers(0,200)
                        ngh.misfolding_level += add_remove
                        agent.misfolding_level-=add_remove
                        ngh.received_misfolding
                        if not self.contains(ngh):
                            new_neuron_spreaders.append(ngh)
        agent_removed=0
        for agent in self.context.agents(Neuron.TYPE):
            self.agent_features.log_row(self.runner.schedule.tick,agent.uid,agent.misfolding_level)
            agent_removed+=agent.step()
        self.agent_features.write()
        self.neuron_spreaders += new_neuron_spreaders
        self.countsNeuron.new_neuron_spreaders = len(new_neuron_spreaders)
        self.countsNeuron.total_neuron_spreaders += self.countsNeuron.new_neuron_spreaders+agent_removed
        self.data_set_diffousori.log(self.runner.schedule.tick)

    def contains(self,a):
        for agent in self.neuron_spreaders:
            if agent.uid[0]==a.uid[0] and agent.uid[2]==a.uid[2]:
                return True
        return False
    def log_counts(self, tick):
        # Get the current number of zombies and humans and log
        self.alpha_protein=0
        self.alpha_mis_protein=0
        self.alpha_oligomer=0
        self.counts.oligome=0
        self.counts.alpha=0
        dead_cells=[]
        self.counts.alphaMis=0
        for a in self.context.agents(Alpha.TYPE):
            if(a.energy<1):
                dead_cells.append(a)
            else:
                if(a.state==0):
                    self.counts.alpha+=1
                    self.alpha_protein+=1
                if(a.state==1):
                    self.counts.alphaMis+=1
                    self.alpha_mis_protein+=1
                if(a.state==2):
                    self.alpha_oligomer+=1
                    self.counts.oligome+=1
        self.data_set.log(tick)
        if(self.rank==0):
            neuron=self.searchAgent(0,1)
            self.alpha_logger.log_row(tick,self.alpha_protein,self.alpha_mis_protein,self.alpha_oligomer)
            if self.searchAgent(0,0):
                agent=self.searchAgent(0,0)
                self.alpha_position.log_row(tick,agent.uid,self.grid.get_location(agent).x,self.grid.get_location(agent).y,agent.state)
        for a in dead_cells:
            self.context.remove(a)
        folding= np.zeros(1, dtype='int64')
        misfolding= np.zeros(1, dtype='int64')
        oligomers=np.zeros(1,dtype='int64')
        self.comm.Reduce(np.array([self.alpha_protein],dtype='int64'),folding,op=MPI.SUM,root=0)
        self.comm.Reduce(np.array([self.alpha_mis_protein],dtype='int64'),misfolding,op=MPI.SUM,root=0)
        self.comm.Reduce(np.array([self.alpha_mis_protein],dtype='int64'),oligomers,op=MPI.SUM,root=0)
        # if(self.rank==2):
        #     print("Tick: {}, Folding Count: {}, Misfolding Count: {}, Oligomer Count: {}".format(tick,self.alpha_protein , self.alpha_mis_protein,self.alpha_oligomer),
        #           flush=True)
    
    def move(self,agent,x,y):
        agent.pt=dpt(x,y,0)
        self.grid.move(agent,dpt(int(math.floor(x)),int(math.floor(y))))

    def at_end(self):
        self.alpha_logger.close()
        self.alpha_position.close()
        self.data_set_diffousori.close()
        self.data_set.close()
        self.neuron_data.close()
        for agent in self.context.agents(Neuron.TYPE):
            print(agent,agent.alpha_synuclein_level,agent.misfolding_level,agent.oligomer_level,agent.lewyBodies_level)
            print(self.grid.get_location(agent))
        # print(self.context.agents())
    def start(self):
        self.runner.execute()

def run(params: Dict):
    global model
    model=Model(MPI.COMM_WORLD,params)
    # generate_network_file("rete",4,20)
    model.start()

if __name__=="__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)
