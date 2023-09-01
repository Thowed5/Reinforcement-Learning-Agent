import numpy as np
import random

class QLearningAgent:
    """
    A Q-learning agent implementation for reinforcement learning tasks.

    Attributes:
        env: The environment the agent interacts with (e.g., OpenAI Gym environment).
        learning_rate (float): The step size for updating Q-values.
        discount_factor (float): The discount factor for future rewards.
        epsilon (float): The exploration-exploitation trade-off parameter.
        q_table (dict): Stores Q-values for state-action pairs.
    """

    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay_rate=0.001, min_epsilon=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.q_table = {}

    def _get_state_key(self, state):
        """
        Converts a state (which can be an array or tuple) into a hashable key for the Q-table.
        """
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state

    def _get_q_value(self, state, action):
        """
        Retrieves the Q-value for a given state-action pair. Initializes to 0 if not seen.
        """
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.env.action_space.n)
        return self.q_table[state_key][action]

    def _set_q_value(self, state, action, value):
        """
        Sets the Q-value for a given state-action pair.
        """
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.env.action_space.n)
        self.q_table[state_key][action] = value

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            state_key = self._get_state_key(state)
            if state_key not in self.q_table:
                return self.env.action_space.sample() # If state not seen, explore
            return np.argmax(self.q_table[state_key]) # Exploit

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-value for the state-action pair using the Q-learning formula.
        """
        current_q = self._get_q_value(state, action)
        next_state_key = self._get_state_key(next_state)

        if done:
            max_next_q = 0
        else:
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.env.action_space.n)
            max_next_q = np.max(self.q_table[next_state_key])

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self._set_q_value(state, action, new_q)

    def decay_epsilon(self):
        """
        Decays the epsilon value over time to reduce exploration.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)


if __name__ == "__main__":
    # Example usage with a dummy environment
    class DummyEnv:
        def __init__(self):
            self.observation_space = type("ObsSpace", (object,), {"n": 2})()
            self.action_space = type("ActSpace", (object,), {"n": 2, "sample": lambda: random.randint(0, 1)})()
            self.state = 0

        def reset(self):
            self.state = 0
            return self.state

        def step(self, action):
            if action == 0: # Move left
                self.state = max(0, self.state - 1)
            else: # Move right
                self.state = min(1, self.state + 1)

            reward = 1 if self.state == 1 else -0.1
            done = self.state == 1
            return self.state, reward, done, {}

    env = DummyEnv()
    agent = QLearningAgent(env, epsilon_decay_rate=0.005)

    num_episodes = 100
    print(f"Training Q-Learning Agent for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")

    print("\nTraining complete. Q-table:")
    for state_key, q_values in agent.q_table.items():
        print(f"State {state_key}: {q_values}")

    print("\nTo use this agent with a real environment (e.g., OpenAI Gym), replace DummyEnv with your desired environment.")
