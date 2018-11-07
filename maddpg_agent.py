from ddpg_agent import DDPGAgent
from networks import Actor


class MADDPG():
    def __init__(self, state_size, action_size, n_agents, random_seed=1):
        self.actor_local = Actor(state_size, action_size, random_seed)
        self.actor_target = Actor(state_size, action_size, random_seed)

        self.ddpg_agents = [DDPGAgent(state_size, action_size, 
                            self.actor_local, self.actor_target, random_seed) for _ in range(n_agents)]

    def act(self, states):
        """Get actions from all agents in the MADDPG object."""
        actions = [agent.act(state) for agent, state in zip(self.ddpg_agents, states)]
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for ddpg_agent, state, action, reward, next_state, done in zip(self.ddpg_agents, states, actions, 
                                                                       rewards, next_states, dones):
            ddpg_agent.step(state, action, reward, next_state, done)

    def get_critic(self):
        """Get the Critic for every agent."""
        return [agent.critic_local for agent in self.ddpg_agents]

    def reset(self):
        for ddpg_agent in self.ddpg_agents:
            ddpg_agent.reset()
