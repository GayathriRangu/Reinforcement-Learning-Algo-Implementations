import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make('LunarLander-v2',render_mode='human')  # continuous: LunarLanderContinuous-v2
env.reset()

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10)

episodes = 10

for ep in range(episodes):
	obs,infor = env.reset()
	print(obs)
	done = False
	while not done:
		# obs=obs.reshape(8,)#reshape to 1 sample
		action, _ = model.predict(obs)
		obs, rewards, done, info, _ = env.step(action)
		env.render()
		print(rewards)
env.close()