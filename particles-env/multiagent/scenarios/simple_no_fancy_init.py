import numpy as np
from multiagent.core import World, Agent, Landmark, BasisPoint
from multiagent.scenario import BaseScenario

NUM_LANDMARKS = 6
BPS = False
NUM_BP = 20

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            #if NUM_LANDMARKS == 11:
            #    agent.size = 0.03
            #agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(NUM_LANDMARKS)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            #landmark.initial_mass = 0.0
            landmark.size = 0.08
            #if NUM_LANDMARKS == 11:
            #    landmark.size = 0.1
        world.landmarks[0].collide = False
        world.landmarks[0].size = 0.05

        bps_pos = np.array([[-0.29374949,  0.50936892],
                             [-0.63307486,  0.29243383],
                             [ 0.1718743,  0.02241032],
                             [-0.82823569,  0.86725642],
                             [-0.64684628, -0.41318022],
                             [ 0.63219611,  0.84687622],
                             [-0.10209109, -0.78531028],
                             [-0.72456915, -0.7982658 ],
                             [ 0.59150568, -0.31033646],
                             [ 0.17624203,  0.20910327],
                             [ 0.42084436, -0.13758224],
                             [-0.97438945, -0.50789116],
                             [-0.64359185,  0.69900274],
                             [ 0.54863065,  0.62152146],
                             [ 0.71214042, -0.59667461],
                             [ 0.91368474,  0.90312691],
                             [ 0.75466108, -0.17432659],
                             [ 0.05915955, -0.58180326],
                             [ 0.34189433, -0.27278016],
                             [ 0.58710656,  0.22424248]])
        if BPS:
            world.bps = [BasisPoint() for i in range(NUM_BP)]
            for i, basis_point in enumerate(world.bps):
                basis_point.name = 'Basis Point %d' % i
                basis_point.state.p_pos = bps_pos[i, :]
                basis_point.color = np.array([1.0, 1.0, 0.0])


        #world.landmarks[1].collide = True
        # make initial conditions
        self.reset_world(world)
        return world

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        eps = 0
        dist_min = agent1.size + agent2.size + eps
        return True if dist <= dist_min else False

    def reset_world(self, world):
        world.timestep = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            #andmark.color = np.array([0.75,0.75,0.75])
            landmark.color = np.array([0.15, 0.15, 0.15])
        world.landmarks[0].color = np.array([0.15, 0.65, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        def collision(i):
            for agent in world.agents:
                if self.is_collision(agent, world.landmarks[i]):
                    return True
            for j in range(i):
                if self.is_collision(world.landmarks[j], world.landmarks[i]):
                    return True
            return False

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            while collision(i):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
        rew -= dist
        #obst_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[1].state.p_pos)))
        #rew += 0.01*obst_dist
        if world.timestep >= 1000:
            rew -= 10
        for i, landmark in enumerate(world.landmarks):
            if self.is_collision(agent, landmark):
                if i == 0:
                    rew += 10
                else:
                    rew -= 10
        #if self.is_collision(world.landmarks[0], agent):
        #    rew = 1.0
        return rew

    def compute_squared_distance(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        return np.sum(np.square(delta_pos))


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        world.timestep += 1
        goal_pos = [world.landmarks[0].state.p_pos - agent.state.p_pos]

        bps_encoding = []
        if BPS:
            #for basis_point in world.bps:
            #    bps_encoding.append(basis_point.state.p_pos - agent.state.p_pos)
            for basis_point in world.bps:
                distances = np.array([self.compute_squared_distance(basis_point, world.landmarks[i]) for i in range(1, NUM_LANDMARKS)])
                closest_obst = np.argmin(distances) + 1
                bps_encoding.append(world.landmarks[closest_obst].state.p_pos - basis_point.state.p_pos)
        else:
            for i, landmark in enumerate(world.landmarks):
                if i == 0:
                    continue
                bps_encoding.append(landmark.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + goal_pos + bps_encoding)
        #return np.concatenate([world.landmarks[0].state.p_pos - agent.state.p_pos] + entity_pos)

    def get_done(self, agent, world):
        if world.timestep >= 1000:
            return 3
        for i, landmark in enumerate(world.landmarks):
            delta_pos = agent.state.p_pos - landmark.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = agent.size + landmark.size
            if dist < dist_min:
                if i==0:
                    return 1
                else:
                    return 2
        return 0