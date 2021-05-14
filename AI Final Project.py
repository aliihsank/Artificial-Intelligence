import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility, helpers, make

from tensorforce import Agent
from tensorforce.agents import PPOAgent, DQNAgent, TRPOAgent, ACAgent, A2CAgent, REINFORCE, DDPG
from tensorforce.execution import Runner
from tensorforce.environments.openai_gym import OpenAIGym
import tensorflow as tf
import datetime

def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


class WrappedEnv(OpenAIGym):    
    def __init__(self, gym, states_spec, actions_spec, visualize=False, max_episode_timesteps=None):
        self.environment = gym
        self.gym = gym
        self.visualize = visualize
        self._max_episode_timesteps = max_episode_timesteps
        self.states_spec = states_spec
        self.actions_spec = actions_spec
    
    def execute(self, actions):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=actions)
            
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward
    
    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
        return agent_obs

# Instantiate the environment
config = ffa_v0_fast_env()
env = Pomme(**config["env_kwargs"])
env.seed(0)

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states=dict(type='float', shape=env.observation_space.shape),
    actions=dict(type='int', num_values=env.action_space.n),
    network='auto',
    max_episode_timesteps = 600,
    batch_size = 100,
    learning_rate=1e-3,
    summarizer=dict(directory="./board5/",
                        #steps=50,
                        summaries = 'all'
                    )
)

# Add 3 SimpleAgents
agents = []
for agent_id in range(3):
    agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

# Add TensorforceAgent
agent_id += 1
agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
env.set_agents(agents)
env.set_training_agent(agents[-1].agent_id)
env.set_init_game_state(None)

# Instantiate and run the environment
wrapped_env = WrappedEnv(env, env.observation_space, env.action_space, True, 600)
runner = Runner(agent=agent, environment=wrapped_env, max_episode_timesteps=600)
runner.run(num_episodes=15000)

# Save agent model
# - format: 'numpy' or 'hdf5' store only weights, 'checkpoint' stores full TensorFlow model
runner.agent.save(directory="C:\\Users\\ali_k\\Desktop\\my_model", format='checkpoint')


# Print resulting stats and graphs
print("Stats: ", runner.episode_rewards, runner.episode_timesteps)

episode_rewards, episode_timesteps = runner.episode_rewards, runner.episode_timesteps

plt.plot(runner.episode_rewards, label='reward')
plt.hist(runner.episode_rewards, label='reward')
plt.plot(runner.episode_timesteps, label='life time')
plt.hist(runner.episode_timesteps, label='life time')

try:
    agent.close()
    runner.close()
except AttributeError as e:
    print("Hata:", e)
    pass


# Test the agent with RandomAgent opponents

test_agents = []
for agent_id in range(3):
    test_agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

# Add TensorforceAgent
agent_id += 1
test_agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
env.set_agents(test_agents)

test_agent = Agent.load(directory="C:\\Users\\ali_k\\Desktop\\my_model", format='checkpoint')

wrapped_env = WrappedEnv(env, env.observation_space, env.action_space, True, 3000)
test_runner = Runner(agent=test_agent, environment=wrapped_env, max_episode_timesteps=2000)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

test_runner.run(num_episodes=100, evaluation=True, evaluation_callback=tensorboard_callback)

# Print test results
plt.plot(test_runner.evaluation_rewards)
plt.plot(test_runner.evaluation_timesteps)


try:
    test_agent.close()
    test_runner.close()
    wrapped_env.close()
except AttributeError as e:
    print("Hata:", e)
    pass






# Test the agent in a new environment

# 200 episode => %9.2
# 300 episode => %9.5
# 500 episode => %6.4 - Bu galiba SimpleAgent'la eğitildi???
# ~13bin episode => %5.9 - RandomAgent
# 10bin episode => % - SimpleAgent'la eğit




