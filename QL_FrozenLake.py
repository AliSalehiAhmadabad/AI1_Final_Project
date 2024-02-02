# AI1 Final Project
# Ali Salehi Ahmadabad

import numpy as np
import gym
from tqdm import tqdm

def get_epsilon_greedy_action(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])

def perform_q_learning(env, num_episodes, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
    Q_values = np.zeros([env.observation_space.n, env.action_space.n])
    progress_bar = tqdm(total=num_episodes, dynamic_ncols=True)
    
    for episode in range(num_episodes):
        current_state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = get_epsilon_greedy_action(Q_values, current_state, exploration_rate)
            next_state, reward, done, _, _ = env.step(action)
            
            best_next_action = np.argmax(Q_values[next_state, :])
            td_target = reward + discount_factor * Q_values[next_state, best_next_action]
            td_error = td_target - Q_values[current_state, action]
            Q_values[current_state, action] += learning_rate * td_error
            
            current_state = next_state
            episode_reward += reward
        
        progress_bar.update(1)
        if episode % 1000 == 0:
            avg_reward = assess_policy(env, Q_values, 100)
            progress_bar.set_description(f"\nAverage reward after {episode} episodes: {avg_reward:.2f}")
    
    progress_bar.close()
    return Q_values

def assess_policy(env, Q, num_episodes):
    total_reward = 0
    policy = np.argmax(Q, axis=1)
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = policy[observation]
            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / num_episodes

def demonstrate_agent(env, Q, num_episodes=1):
    policy = np.argmax(Q, axis=1)
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        print("\nEpisode:", episode + 1)
        
        while not done:
            env.render()
            action = policy[observation]
            observation, _, done, _, _ = env.step(action)
        
        env.render()

def main():
    environment = gym.make("FrozenLake-v1")
    num_episodes = 10000

    learned_Q = perform_q_learning(environment, num_episodes)
    average_reward = assess_policy(environment, learned_Q, num_episodes)
    print(f"Average reward after Q-learning: {average_reward}")

    visual_environment = gym.make('FrozenLake-v1', render_mode='human')
    demonstrate_agent(visual_environment, learned_Q, 10)

if __name__ == '__main__':
    main()
