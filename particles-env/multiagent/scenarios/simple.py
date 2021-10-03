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
            if NUM_LANDMARKS == 11:
                agent.size = 0.03

        # add landmarks
        world.landmarks = [Landmark() for i in range(NUM_LANDMARKS)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.08
            if NUM_LANDMARKS == 11:
                landmark.size = 0.1
        world.landmarks[0].collide = False
        world.landmarks[0].size = 0.05

        if BPS:
            world.bps = [BasisPoint() for i in range(NUM_BP)]
            for i, basis_point in enumerate(world.bps):
                basis_point.name = 'Basis Point %d' % i
                basis_point.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
                basis_point.color = np.array([1.0, 1.0, 0.0])

        # make initial conditions
        self.reset_world(world)
        return world

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        eps = 0.01
        dist_min = agent1.size + agent2.size + eps
        return True if dist <= dist_min else False

    def reset_world(self, world):
        world.timestep = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        world.landmarks[0].color = np.array([0.15, 0.65, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        def collision(i):
            if self.is_collision(world.agents[0], world.landmarks[i]):
                return True
            for j in range(i):
                if self.is_collision(world.landmarks[j], world.landmarks[i]):
                    return True
            return False

        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                while np.sqrt(self.compute_squared_distance(landmark, world.agents[0])) < 1.0:
                    landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

            else:
                vec_agent_goal = world.landmarks[0].state.p_pos - world.agents[0].state.p_pos
                landmark.state.p_pos = np.random.uniform(0,+1) * vec_agent_goal + world.agents[0].state.p_pos + np.random.normal(0.0, 0.15, world.dim_p)
                while collision(i):
                    landmark.state.p_pos = np.random.uniform(0,+1) * vec_agent_goal + world.agents[0].state.p_pos + np.random.normal(0.0, 0.15, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
        rew -= dist
        if world.timestep >= 1000:
            rew -= 10
        for i, landmark in enumerate(world.landmarks):
            if self.is_collision(agent, landmark):
                if i == 0:
                    rew += 10
                else:
                    rew -= 10
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
            for basis_point in world.bps:
                bps_encoding.append(basis_point.state.p_pos - agent.state.p_pos)
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

    def get_done(self, agent, world):
        if world.timestep >= 1000:
            return 2
        for i, landmark in enumerate(world.landmarks):
            delta_pos = agent.state.p_pos - landmark.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = agent.size + landmark.size
            if dist < dist_min:
                if i == 0:
                    return 1
                else:
                    return 2
        return 0
