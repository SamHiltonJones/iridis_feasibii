from stable_baselines3 import PPO
import torch as th
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

env_name = r"C:\Users\samhi\scene1_builds\3DPos.exe"

channel = EngineConfigurationChannel()
unity_env = UnityEnvironment(env_name, side_channels=[channel])
channel.set_configuration_parameters(time_scale = 1)
env = UnityToGymWrapper(unity_env)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[128, 256, 256, 128], vf=[128, 256, 256, 128])])

model = PPO.load(r"trained_models\ppo_model_scene1_07082024_v0.zip", env=env)

model.set_env(env)

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print("Act: " + str(action))
    print("Obs : " + str(obs))
    print(rewards)
    if dones:
        obs = env.reset()
    env.render()