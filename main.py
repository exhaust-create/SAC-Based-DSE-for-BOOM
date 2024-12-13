# coding=utf-8
import random
import matplotlib.pyplot as plt
import torch
import yaml
from typing import Dict
import os, sys
import numpy as np
import multiprocessing
import pandas as pd
from functools import partial

os.chdir(sys.path[0])

from env.env_no_option import BOOMENV as Env_new
from env.env_with_option import BOOMENV as HRLEnv
from drawer.PlotLearningCurve import plot_result, ema_plotting
from PPO.PPO2 import PPO2_v0, PPO2_v1
from SAC.SAC import SAC
from Comparisons.MoEnDSE import BagGBRT
from Comparisons.MyDatasetCreation.problem import DesignSpaceProblem

# For test
from torch.distributions import Categorical
import timeit

start = timeit.default_timer()

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)


# Random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Read the config.yml file
def get_configs(fyaml: str) -> Dict:
    with open(fyaml, 'r') as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            configs = yaml.load(f)
    return configs


def ppov0_run(config, seed=10):
    times = 10
    max_episodes = config["max_episodes"] * times
    num_seed = 3
    all_rewards = np.zeros((config["max_episodes"], num_seed))  # (num_episode, num_seeds)
    indices = np.arange(0, max_episodes, times)
    mean_episode_rewards, std_episode_rewards = [], []
    entropy = []
    for j in range(num_seed):
        setup_seed(seed + j)
        env = HRLEnv(config, device)
        agent = PPO2_v0(config, env, device)
        total_rewards, episodes, final_microarchs, norm_ppas = [], [], [], []

        # Run a whole process with a random seed.
        for i in range(max_episodes):
            agent.run_an_episode()
            total_reward = env.get_cum_reward()
            total_rewards.append(total_reward)  # Can only depict the distribution of the values.
            final_microarchs.append(env._explored_designs[-1][0])
            norm_ppas.append(env._explored_designs[-1][1])
            print('EP:{} total_reward:{}'.
                  format(i + 1, total_reward))
        entropy.append(np.array(agent.entropy))
        all_rewards[:, j] = np.array(total_rewards).squeeze(-1)[indices]

        best_design_episode, best_design, best_design_ppa, proj = env.get_best_point(np.array(final_microarchs), np.array(norm_ppas),
                                                                np.array(total_rewards))
        print("best_episode: {}, best_design: {}, best_design_ppa: {}, projection: {}".format(best_design_episode, best_design, best_design_ppa, proj))

    # Get the means and standard deviations of episode rewards.
    # These two are what should be plotted.
    mean = all_rewards.mean(axis=1)
    std = all_rewards.std(axis=1)
    # for i in range(len(all_rewards)):
    #     mean_episode_rewards.append(mean[i])
    #     std_episode_rewards.append(std[i])

    # Plot
    # total_reward_curve_file_path = os.path.join(config["reports_folder_path"], "ppov0_total_reward_3width_721.pdf")
    # avg_reward_curve_file_path = os.path.join(config["reports_folder_path"], "ppov0_avg_reward_1width_361.pdf")
    # ema_num_episodes = [i for i in range(len(all_rewards))]
    # # episodes = [i for i in range(max_episodes)]
    # ema_plotting(ema_num_episodes, mean_episode_rewards, 10, 'Avg_Reward', 'avg_reward', avg_reward_curve_file_path, std=std_episode_rewards)
    # plot_result(episodes, total_rewards, 'Total_Reward', 'total_reward', total_reward_curve_file_path)
    return mean, std, entropy


def ppov1_run(config, seed=10):
    max_episodes = config["max_episodes"]
    num_seed = 3
    all_rewards = np.zeros((max_episodes, num_seed))  # (num_episode, num_seeds)
    mean_episode_rewards, std_episode_rewards = [], []
    entropy = []
    for j in range(num_seed):
        setup_seed(seed + j)
        env = Env_new(config, device)
        agent = PPO2_v1(config, env, device, embedding=True)  # Emperiments show that embedding is essential.
        total_rewards, episodes, final_microarchs, norm_ppas = [], [], [], []

        # Run a whole process with a random seed.
        for i in range(max_episodes):
            agent.run_an_episode()
            total_reward = env.get_cum_reward()
            all_rewards[i, j] = total_reward.copy()
            total_rewards.append(total_reward)
            final_microarchs.append(env._explored_designs[-1][0])
            norm_ppas.append(env._explored_designs[-1][1])
            print('EP:{} total_reward:{}'.
                  format(i + 1, total_reward))

        entropy.append(np.array(agent.entropy))

        best_design_episode, best_design, best_design_ppa, proj = env.get_best_point(np.array(final_microarchs), np.array(norm_ppas),
                                                                np.array(total_rewards))
        print("best_episode: {}, best_design: {}, best_design_ppa: {}, projection: {}".format(best_design_episode, best_design, best_design_ppa, proj))

    # Get the means and standard deviations of episode rewards.
    mean = all_rewards.mean(axis=1)
    std = all_rewards.std(axis=1)
    # for i in range(max_episodes):
    #     mean_episode_rewards.append(mean[i])
    #     std_episode_rewards.append(std[i])

    # # Plot
    # total_reward_curve_file_path = os.path.join(config["reports_folder_path"], "ppov1_total_reward_3width_361.pdf")
    # avg_reward_curve_file_path = os.path.join(config["reports_folder_path"], "ppov1_avg_reward_3width_361.pdf")
    # episodes = [i for i in range(max_episodes)]
    # # ema_plotting(episodes, total_rewards, 10, 'Avg_Reward', 'avg_reward', avg_reward_curve_file_path)
    # ema_plotting(episodes, mean_episode_rewards, 10, 'Avg_Reward', 'avg_reward', avg_reward_curve_file_path, std=std_episode_rewards)
    # plot_result(episodes, total_rewards, 'Total_Reward', 'total_reward', total_reward_curve_file_path)
    return mean, std, entropy


def sac_run(config, seed=10):
    max_episodes = config["max_episodes"]  # 80 episodes is enough for convergence.
    num_seed = 3
    all_rewards = np.zeros((max_episodes, num_seed))  # (num_episode, num_seeds)
    mean_episode_rewards, std_episode_rewards, entropy = [], [], []
    for j in range(num_seed):
        actual_seed = seed + j
        setup_seed(actual_seed)
        env = Env_new(config, device)
        agent = SAC(config, env, device,
                    embedding=True)  # Embedding will decrease the num of training data to some extent.
        total_rewards, episodes, final_microarchs, norm_ppas = [], [], [], []

        # Run a whole process with a random seed.
        for i in range(max_episodes):
            agent.run_an_episode()
            total_reward = env.get_cum_reward()
            all_rewards[i, j] = total_reward.copy()
            total_rewards.append(total_reward)
            final_microarchs.append(env._explored_designs[-1][0])
            norm_ppas.append(env._explored_designs[-1][1])
            print('EP:{} total_reward:{}'.
                  format(i + 1, total_reward))

        actor_loss = np.array(agent.actor_loss).reshape(1, -1)
        critic_1_loss = np.array(agent.critic_1_loss).reshape(1,-1)
        critic_2_loss = np.array(agent.critic_2_loss).reshape(1,-1)
        loss = np.concat((actor_loss, critic_1_loss, critic_2_loss), axis=0)
        label = ["actor_loss", "critic_1_loss", "critic_2_loss"]
        target_value = np.array(agent.target_value).reshape(1, -1)
        target_value_label = ["target_value"]
        entropy.append(np.array(agent.entropy))

        best_design_episode, best_design, best_design_ppa, proj = env.get_best_point(np.array(final_microarchs), np.array(norm_ppas),
                                                                np.array(total_rewards))
        print("best_episode: {}, best_design: {}, best_design_ppa: {}, projection: {}".format(best_design_episode, best_design, best_design_ppa, proj))
        # length = np.arange(critic_1_loss.shape[-1])
        # plot_result(length, loss, 'Loss', 'loss', os.path.join(config["reports_folder_path"],"SAC_loss_seed"+str(actual_seed)+".pdf"), label=label, dot=False)
        # state_length = np.arange(target_value.shape[-1])
        # ema_plotting(state_length, target_value, 15, "Target Value", "target_value", os.path.join(config["reports_folder_path"],"SAC_target_value_seed"+str(actual_seed)+".pdf"), label=target_value_label)

    # Get the means and standard deviations of episode rewards.
    mean = all_rewards.mean(axis=1)
    std = all_rewards.std(axis=1)
    # for i in range(max_episodes):
    #     mean_episode_rewards.append(mean[i])
    #     std_episode_rewards.append(std[i])

    # Plot
    # total_reward_curve_file_path = os.path.join(config["reports_folder_path"], "sac_total_reward_1width_361_action_rand.pdf")
    # avg_reward_curve_file_path = os.path.join(config["reports_folder_path"], "sac_avg_reward_1width_361_action_rand.pdf")
    # episodes = np.array([i for i in range(max_episodes)])
    # # ema_plotting(episodes, total_rewards, 10, 'Avg_Reward', 'avg_reward', avg_reward_curve_file_path)
    # ema_plotting(episodes, mean, 10, 'Avg_Reward', 'avg_reward', avg_reward_curve_file_path, label=["sac"], std=std)
    # plot_result(episodes, total_rewards, 'Total_Reward', 'total_reward', total_reward_curve_file_path)
    return mean, std, entropy


# def BagGBRT_run(config, seed=10):
#     max_episodes = config["max_episodes"]
#     times = 10
#     train_iter = times*max_episodes
#     num_seed = 3
#     all_rewards = np.zeros((max_episodes+1, num_seed))   # (num_episode, num_seeds)
#     indices = np.arange(0,train_iter,times)
#     mean_episode_rewards, std_episode_rewards = [], []
#     for j in range(num_seed):
#         setup_seed(seed + j)
#         problem = DesignSpaceProblem(config)
#         agent = BagGBRT(problem)
#         found_designs, found_designs_ppa, found_designs_proj = agent.train(train_iter)

#         all_rewards[:-1,j] = np.array(found_designs_proj)[indices]
#         all_rewards[-1,j] = found_designs_proj[-1]

#         best_design, best_design_ppa, proj = agent.get_best_point(np.array(found_designs), np.array(found_designs_ppa), np.array(found_designs_proj))
#         print("best_design: {}, best_design_ppa: {}, projection: {}".format(best_design, best_design_ppa, proj))

#     # Get the means and standard deviations of episode rewards.
#     mean = all_rewards.mean(axis=1)
#     std = all_rewards.std(axis=1)
#     for i in range(len(all_rewards)):
#         mean_episode_rewards.append(mean[i])
#         std_episode_rewards.append(std[i])

#     # Plot
#     avg_reward_curve_file_path = os.path.join(config["reports_folder_path"], "BagGBRT_avg_reward_1width_361.pdf")
#     ema_num_episodes = [i for i in range(len(all_rewards))]
#     # episodes = [i for i in range(max_episodes)]
#     ema_plotting(ema_num_episodes, mean_episode_rewards, 10, 'Avg_Reward', 'avg_reward', avg_reward_curve_file_path, std=std_episode_rewards)


# ---------------- BagGBRT Multiprocessing ---------------- #
# BagGBRT runs very slowly, so I split it into two functions for multiprocessing.
def baggbrt_run_once(config, seed=10):
    max_episodes = config["max_episodes"]
    times = 10
    train_iter = times * max_episodes
    all_rewards = np.zeros((max_episodes, 1))  # (num_episode, num_seeds)
    indices = np.arange(0, train_iter, times)

    setup_seed(seed)
    problem = DesignSpaceProblem(config)
    agent = BagGBRT(problem)
    found_designs, found_designs_ppa, found_designs_proj = agent.train(
        train_iter)  # "found_designs_ppa" is normalized PPA.
    
    # train_mse = np.array(agent.train_mse).reshape(1,-1)
    test_mse = np.array(agent.test_mse)

    all_rewards[:, 0] = np.array(found_designs_proj)[indices]

    best_design_episode, best_design, best_design_ppa, proj = agent.get_best_point(np.array(found_designs), np.array(found_designs_ppa),
                                                              np.array(found_designs_proj))
    print("best_episode: {}, best_design: {}, best_design_ppa: {}, projection: {}".format(best_design_episode, best_design, best_design_ppa, proj))
    return np.array(found_designs_proj), all_rewards, test_mse


def baggbrt_run_multiprocess(config, seed):
    num_seed = 3
    seeds = [(seed + i) for i in range(num_seed)]
    mean_episode_rewards, std_episode_rewards = [], []

    func = partial(baggbrt_run_once, config)
    with multiprocessing.Pool(3) as pool:
        results = pool.map(func, seeds)
    found_designs_projs, all_rewards, test_mse = zip(*results)
    all_rewards = np.array(all_rewards).squeeze(-1).transpose()
    total_rewards = np.array(found_designs_projs)[-1].transpose().tolist()  # All found designs at the last trial.
    # mse = np.concat((train_mse, test_mse), axis=0)

    # Get the means and standard deviations of episode rewards.
    mean = all_rewards.mean(axis=1)
    std = all_rewards.std(axis=1)
    # for i in range(len(all_rewards)):
    #     mean_episode_rewards.append(mean[i])
    #     std_episode_rewards.append(std[i])

    # Plot
    # total_reward_curve_file_path = os.path.join(config["reports_folder_path"], "BagGBRT_total_reward_3width_361.pdf")
    # avg_reward_curve_file_path = os.path.join(config["reports_folder_path"], "BagGBRT_avg_reward_3width_361.pdf")
    # ema_num_episodes = [i for i in range(len(all_rewards))]
    # episodes = [i for i in range(len(total_rewards))]
    # # episodes = [i for i in range(max_episodes)]
    # ema_plotting(ema_num_episodes, mean_episode_rewards, 10, 'Avg_Reward', 'avg_reward', avg_reward_curve_file_path, std=std_episode_rewards)
    # plot_result(episodes, total_rewards, 'Total_Reward', 'total_reward', total_reward_curve_file_path)
    return mean, std, test_mse
# ---------------- BagGBRT Multiprocessing END---------------- #

if __name__ == '__main__':
    # Read Width and Preference
    parser = argparse.ArgumentParser()
    parser.add_argument('--width_pref', default = None)
    args = parser.parse_args()
    
    start = timeit.default_timer()

    seed = 300
    width_pref = "5W_721" if args.width_pref is None else args.width_pref  # Change this to modify the names of "config" files and csv file.
    config_sac = get_configs("config/config_sac_"+width_pref+".yml")
    config_ppo = get_configs("config/config_ppo_"+width_pref+".yml")

    baggbrt_mean, baggbrt_std, test_mse = baggbrt_run_multiprocess(config_sac, seed)
    ppov0_mean, ppov0_std, ppov0_entropy = ppov0_run(config_ppo, seed)
    ppov1_mean, ppov1_std, ppov1_entropy = ppov1_run(config_ppo, seed)
    sac_mean, sac_std, sac_entropy = sac_run(config_sac, seed)

    end = timeit.default_timer()
    print('Running time: %s Seconds' % (end - start))

    # Plot Rewards
    episodes = np.array([i for i in range(len(sac_mean))])
    mean = np.concatenate((ppov0_mean.reshape(1, -1), ppov1_mean.reshape(1, -1), sac_mean.reshape(1, -1), baggbrt_mean.reshape(1, -1)), axis=0)
    std = np.concatenate((ppov0_std.reshape(1, -1), ppov1_std.reshape(1, -1), sac_std.reshape(1, -1), baggbrt_std.reshape(1, -1)), axis=0)
    label=['ppov0','ppov1','sac','baggbrt']
    ema_plotting(episodes, mean, 15, 'Avg_Reward', 'avg_reward', os.path.join(config_sac["reports_folder_path"], config_sac["avg_reward_curve_path"]), label=label)
    
    # # Save all mean and std values.
    # data={
    #     'ppov0_mean': ppov0_mean,
    #     'ppov0_std': ppov0_std,
    #     'ppov1_mean': ppov1_mean,
    #     'ppov1_std': ppov1_std,
    #     'sac_mean': sac_mean,
    #     'sac_std': sac_std,
    #     'baggbrt_mean': baggbrt_mean,
    #     'baggbrt_std': baggbrt_std
    # }
    # pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "mean_&_std_"+width_pref+".csv"), index=False)

    # Plot MSE
    mse_label = ["trial 1", "trial 2", "trial 3"]
    mse_length = np.array([i for i in range(test_mse[0].shape[-1])])
    plot_result(mse_length, test_mse, None, 'mse', os.path.join(config_sac["reports_folder_path"],"Entropy/MoEnDSE_"+str(width_pref)+".pdf"), label=mse_label, xlabel='exploration rounds', dot=False)

    # Plot SAC Entropy
    entropy_label = ["trial 1", "trial 2", "trial 3"]
    max_num_entropy = max([len(sac_entropy[i]) for i in range(len(sac_entropy))])
    entropy_length = np.arange(max_num_entropy)
    plot_result(entropy_length, sac_entropy, None, "entropy", os.path.join(config_sac["reports_folder_path"], "Entropy/SAC_entropy_"+str(width_pref)+".pdf"), label=entropy_label, xlabel="update times", dot=False)

    # Plot PPO_v0 Entropy
    max_num_entropy = max([len(ppov0_entropy[i]) for i in range(len(ppov0_entropy))])
    entropy_length = np.arange(max_num_entropy)
    plot_result(entropy_length, ppov0_entropy, None, "entropy", os.path.join(config_sac["reports_folder_path"], "Entropy/PPOv0_entropy_"+str(width_pref)+".pdf"), label=entropy_label, xlabel="update times", dot=False)

    # Plot PPO_v1 Entropy
    max_num_entropy = max([len(ppov1_entropy[i]) for i in range(len(ppov1_entropy))])
    entropy_length = np.arange(max_num_entropy)
    plot_result(entropy_length, ppov1_entropy, None, "entropy", os.path.join(config_sac["reports_folder_path"], "Entropy/PPOv1_entropy_"+str(width_pref)+".pdf"), label=entropy_label, xlabel="update times", dot=False)

    # # For test
    # mean = np.concatenate((ppov1_mean.reshape(1, -1), sac_mean.reshape(1, -1)), axis=0)
    # label=['ppov1','sac']
    # episodes = np.array([i for i in range(len(sac_mean))])
    # ema_plotting(episodes, mean, 15, 'Avg_Reward', 'avg_reward', os.path.join(config_sac["reports_folder_path"], config_sac["avg_reward_curve_path"]), label=label)
    # data={
    #     'ppov1_mean': ppov1_mean,
    #     'ppov1_std': ppov1_std,
    #     'sac_mean': sac_mean,
    #     'sac_std': sac_std
    # }
    # pd.DataFrame(data).to_csv(os.path.join(config_sac["reports_folder_path"], "ppov1_sac_"+width_pref+".csv"), index=False)
