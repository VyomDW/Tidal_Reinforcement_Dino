import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class VSCodeDinoEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(2) # 0 = Run, 1 = Jump
        self.observation_space = spaces.Box(
            low=np.array([0.0, -15.0, -50.0, 0.0, 4.0], dtype=np.float32),
            high=np.array([150.0, 15.0, 600.0, 60.0, 20.0], dtype=np.float32),
            dtype=np.float32
        )
        self.render_mode = render_mode
        self.width, self.height = 600, 150
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("TIDAL RL Workshop - Chrome Dino")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 14)
            
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dino_x, self.dino_y, self.dino_vel_y = 40.0, 0.0, 0.0
        self.gravity, self.jump_strength, self.is_jumping = 0.8, 10.5, False
        self.speed, self.obstacle_x, self.obstacle_width, self.obstacle_height = 6.0, 550.0, 20.0, 35.0
        self.score = 0
        return self._get_obs(), {}

    def _get_obs(self):
        dist = self.obstacle_x - (self.dino_x + 20.0)
        return np.array([self.dino_y, self.dino_vel_y, dist, self.obstacle_width, self.speed], dtype=np.float32)

    def step(self, action):
        self.score += 1
        reward = 0.1
        if action == 1 and not self.is_jumping:
            self.is_jumping, self.dino_vel_y = True, self.jump_strength

        if self.is_jumping:
            self.dino_y += self.dino_vel_y
            self.dino_vel_y -= self.gravity
            if self.dino_y <= 0.0:
                self.dino_y = self.dino_vel_y = 0.0
                self.is_jumping = False

        self.obstacle_x -= self.speed
        if self.obstacle_x < -30.0:
            self.obstacle_x = random.uniform(480.0, 620.0)
            self.obstacle_width = random.choice([20.0, 30.0, 40.0])
            self.speed = min(self.speed + 0.05, 14.0)
            reward += 2.0

        terminated = False
        if (self.dino_x + 24.0 > self.obstacle_x and self.dino_x < self.obstacle_x + self.obstacle_width):
            if self.dino_y < self.obstacle_height - 5.0:
                terminated, reward = True, -20.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, self.score >= 5000, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((250, 250, 250))
        # Ground Line
        pygame.draw.line(self.screen, (100, 100, 100), (0, 120), (self.width, 120), 2)
        # Cactus
        pygame.draw.rect(self.screen, (40, 140, 40), (int(self.obstacle_x), int(120 - self.obstacle_height), int(self.obstacle_width), int(self.obstacle_height)))
        # Dino
        pygame.draw.rect(self.screen, (60, 60, 60), (int(self.dino_x), int(120 - 32 - self.dino_y), 24, 32))
        pygame.draw.rect(self.screen, (255, 255, 255), (int(self.dino_x) + 16, int(120 - 28 - self.dino_y), 3, 3))
        # Score
        txt = self.font.render(f"Score: {self.score}", True, (80, 80, 80))
        self.screen.blit(txt, (self.width - 100, 10))

        pygame.display.flip()
        self.clock.tick(60) # 60 FPS smooth playback

#Execution 
if __name__ == "__main__":
    print("1. Training Dino AI Agent (~15 seconds)...")
    train_env = VSCodeDinoEnv(render_mode=None)
    model = PPO("MlpPolicy", train_env, verbose=0, learning_rate=0.001, n_steps=1024)
    model.learn(total_timesteps=45000)
    print("   Training Complete!")

    print("\n2. Launching Live Game Window...")
    eval_env = VSCodeDinoEnv(render_mode="human")
    obs, _ = eval_env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        if terminated or truncated:
            obs, _ = eval_env.reset()