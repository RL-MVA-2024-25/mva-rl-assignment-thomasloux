import os
import random
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
try:
    from fast_env_py import FastHIVPatient as HIVPatient
except ImportError:
    from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
import argparse


def make_env(domain_randomization=True):
    return lambda: TimeLimit(HIVPatient(domain_randomization=domain_randomization), max_episode_steps=200)


def linear_schedule(a, b):
    return lambda p: b + (a - b) * p


class ProjectAgent:
    def __init__(self):
        pk = {'net_arch': [{
            'pi': [256, 256],
            'vf': [512, 512]
            }]}
        domain_randomization = True
        self.env = VecNormalize(SubprocVecEnv([make_env(domain_randomization) for _ in range(8)]), norm_obs=True, norm_reward=True, clip_obs=10)
        self.model = PPO(MlpPolicy, self.env, verbose=1, policy_kwargs=pk, learning_rate=linear_schedule(3e-3,1e-3))
    
    def load(self, m_path="model.zip", e_path="vec_normalize.pkl"):
        if os.path.exists(m_path):
            self.model = PPO.load(m_path, env=self.env)
            print(f"Loaded model from {m_path}")
        else:
            print(f"Model file {m_path} not found.")
            return
        if os.path.exists(e_path):
            self.env = VecNormalize.load(e_path, self.env)
            self.env.training = False
            self.env.norm_reward = False
        else: 
            print(f"Env file {e_path} not found.")
    
    def train(self, t=5_000):
        self.model.learn(total_timesteps=t)
        self.save()

    def act(self, obs, rand=False):
        if rand:
            return random.randint(0, 3)
        else:
            return self.model.predict(self.env.normalize_obs(obs), deterministic=True)[0]
    
    def save(self, m_path="model.zip", e_path="vec_normalize.pkl"):
        self.model.save(m_path)
        print(f"Model saved to {m_path}")
        self.env.save(e_path)
    
    def close(self):
        self.env.close()

    def evaluate(self):
        reward = evaluate_HIV(self)
        reward_population = evaluate_HIV_population(self)
        print(f"Reward: {reward}, Reward Population: {reward_population}")
        with open("score.txt", "w") as f:
            f.write(f"{reward}\n{reward_population}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    agent = ProjectAgent()
    if args.retrain:
        print("Retraining agent")
        agent.load()
        agent.train(100_000)

    if args.train:
        print("Training agent")
        # agent = ProjectAgent()
        agent.train(1_000_000)
        # agent.train(20_000)
    agent.load()
    agent.evaluate()
    agent.close()

if __name__ == "__main__":
    main()