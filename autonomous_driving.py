import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class DrivingAgent:
    def __init__(self, state_shape, n_actions, device="cpu"):
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Define discrete actions
        self.actions = [
            [-1.0, 0.0, 0.0],  # Full left
            [1.0, 0.0, 0.0],   # Full right
            [0.0, 1.0, 0.0],   # Full gas
            [0.0, 0.0, 0.8],   # Full brake
            [0.0, 0.0, 0.0],   # No action
        ]
        
        self.model = DQN(state_shape, len(self.actions)).to(device)
        self.target_model = DQN(state_shape, len(self.actions)).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_action(self, action_idx):
        return np.array(self.actions[action_idx], dtype=np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

def process_state(state):
    """Convert state to the correct format for the neural network"""
    state = np.transpose(state, (2, 0, 1))  # Convert to channel-first format
    state = state.astype(np.float32) / 255.0  # Normalize pixel values
    return state

def train_agent(episodes=1000, max_steps=1000):
    env = gym.make('CarRacing-v3', render_mode="human", domain_randomize=False)
    state_shape = (3, 96, 96)  # RGB image
    n_actions = 5  # Number of discrete actions we defined
    
    agent = DrivingAgent(state_shape, n_actions)
    best_reward = float('-inf')
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            state = process_state(state)
            total_reward = 0
            
            for step in range(max_steps):
                action_idx = agent.act(state)
                action = agent.get_action(action_idx)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_state = process_state(next_state)
                
                agent.remember(state, action_idx, reward, next_state, done)
                agent.replay()
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(agent.model.state_dict(), "best_model.pth")
            
            if episode % 10 == 0:
                agent.update_target_model()
                print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Best Reward: {best_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            
            if episode % 100 == 0:
                torch.save(agent.model.state_dict(), f"model_checkpoint_{episode}.pth")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final model...")
        torch.save(agent.model.state_dict(), "interrupted_model.pth")
    
    finally:
        env.close()
    
    return agent

if __name__ == "__main__":
    trained_agent = train_agent()