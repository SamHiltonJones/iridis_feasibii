from stable_baselines3 import TD3
import torch as th
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

env_name = r"C:\Users\samhi\scene3_builds\3DPos.exe"

channel = EngineConfigurationChannel()
unity_env = UnityEnvironment(env_name, side_channels=[channel])
channel.set_configuration_parameters(time_scale=1)
env = UnityToGymWrapper(unity_env)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[256, 256, 256, 256])

model = TD3.load(r"trained_models\td3_model_scene2.zip", env=env)

model.set_env(env)

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print("Act: " + str(action))
    print("Obs: " + str(obs))
    print("Rewards: " + str(rewards))
    if dones:
        obs = env.reset()
    env.render()
