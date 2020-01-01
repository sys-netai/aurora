import ray
import gym
import multiprocessing
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from network_sim import SimulatedNetworkEnv

def train():
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["env_config"] = {
        "history_len": 10,
        "features": "sent latency inflation,latency ratio,send ratio"
    }
    config["num_workers"] = 6
    config["eager"] = False
    config["log_level"] = "INFO"
    config["monitor"] = True
    config["num_cpus_per_worker"] = 0
    trainer = ppo.PPOTrainer(config=config, env=SimulatedNetworkEnv)

    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)


if __name__ == "__main__":
    train()
