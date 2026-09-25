import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ---------------------------------------------------------
# Pixel Art Drawing Utilities for Chrome Dino & Cacti
# ---------------------------------------------------------
def draw_pixel_dino(surface, x, y, frame=0, is_jumping=False):
    """Draws a crisp, animated Chrome Dino using a pixel grid."""
    pixel_size = 2
    color = (83, 83, 83) # Classic Chrome Dino dark grey
    
    # 22x24 grid representation of the Chrome Dino body & head
    body = [
        "          ########## ",
        "          # ######## ",
        "          ########## ",
        "          ########## ",
        "          ########## ",
        "          ###        ",
        "          ######     ",
        "#        #######     ",
        "#        #######     ",
        "##      ########     ",
        "###    #########     ",
        "####  ##########     ",
        "################     ",
        " ###############     ",
        "  ##############     ",
        "   #############     ",
        "    ###########      ",
        "     #########       ",
        "      #######        ",
    ]
    
    for row_idx, row in enumerate(body):
        for col_idx, char in enumerate(row):
            if char == '#':
                px = x + col_idx * pixel_size
                py = y + row_idx * pixel_size
                pygame.draw.rect(surface, color, (px, py, pixel_size, pixel_size))

    # Eye (white pixel)
    pygame.draw.rect(surface, (255, 255, 255), (x + 12 * pixel_size, y + 1 * pixel_size, pixel_size, pixel_size))

    # Animated Legs
    leg_y = y + len(body) * pixel_size
    if is_jumping:
        # Tucked legs
        pygame.draw.rect(surface, color, (x + 7 * pixel_size, leg_y, 2 * pixel_size, 3 * pixel_size))
        pygame.draw.rect(surface, color, (x + 11 * pixel_size, leg_y, 2 * pixel_size, 3 * pixel_size))
    else:
        # Running animation (alternate legs)
        if frame % 2 == 0:
            # Left leg down, right leg up
            pygame.draw.rect(surface, color, (x + 7 * pixel_size, leg_y, 2 * pixel_size, 5 * pixel_size))
            pygame.draw.rect(surface, color, (x + 11 * pixel_size, leg_y, 2 * pixel_size, 2 * pixel_size))
        else:
            # Left leg up, right leg down
            pygame.draw.rect(surface, color, (x + 7 * pixel_size, leg_y, 2 * pixel_size, 2 * pixel_size))
            pygame.draw.rect(surface, color, (x + 11 * pixel_size, leg_y, 2 * pixel_size, 5 * pixel_size))


def draw_pixel_cactus(surface, x, y, width, height):
    """Draws pixelated cactus obstacles."""
    color = (83, 83, 83)
    # Main trunk
    pygame.draw.rect(surface, color, (x + width // 3, y, width // 3, height))
    # Left arm
    pygame.draw.rect(surface, color, (x, y + height // 3, width // 3, height // 3))
    pygame.draw.rect(surface, color, (x, y + height // 4, width // 4, height // 3))
    # Right arm
    pygame.draw.rect(surface, color, (x + (2 * width) // 3, y + height // 3, width // 3, height // 3))
    pygame.draw.rect(surface, color, (x + (3 * width) // 4, y + height // 4, width // 4, height // 3))


# ---------------------------------------------------------
# 1. Solidified Environment with Animated Visuals
# ---------------------------------------------------------
class ChromeDinoEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(2) # 0 = Run, 1 = Jump
        
        # State: [Dino Y, Dino Y-Vel, Distance to Next Obstacle, Obstacle Width, Game Speed]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -15.0, -100.0, 0.0, 4.0], dtype=np.float32),
            high=np.array([150.0, 15.0, 650.0, 60.0, 22.0], dtype=np.float32),
            dtype=np.float32
        )
        self.render_mode = render_mode
        self.width, self.height = 600, 150
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("TIDAL RL Workshop - Chrome Dino RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 14)
            
        self.anim_frame = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dino_x = 40.0
        self.dino_y = 0.0
        self.dino_w = 44.0
        self.dino_h = 48.0
        self.dino_vel_y = 0.0
        self.gravity = 0.85
        self.jump_strength = 11.5
        self.is_jumping = False
        
        self.speed = random.uniform(5.5, 7.0)
        self.obstacle_x = random.uniform(500.0, 650.0)
        self.obstacle_width = random.choice([24.0, 36.0, 48.0])
        self.obstacle_height = 40.0
        
        self.score = 0
        self.anim_frame = 0
        return self._get_obs(), {}

    def _get_obs(self):
        dist = self.obstacle_x - (self.dino_x + self.dino_w)
        return np.array([self.dino_y, self.dino_vel_y, dist, self.obstacle_width, self.speed], dtype=np.float32)

    def step(self, action):
        self.score += 1
        reward = 0.1
        dist = self.obstacle_x - (self.dino_x + self.dino_w)

        # Toggle leg animation frame every 5 game steps
        if self.score % 5 == 0:
            self.anim_frame = (self.anim_frame + 1) % 2

        # Jump Logic & Early Jump Penalty
        if action == 1 and not self.is_jumping:
            self.is_jumping = True
            self.dino_vel_y = self.jump_strength
            if dist > 180.0: 
                reward -= 0.5

        # Gravity & Physics Arc
        if self.is_jumping:
            self.dino_y += self.dino_vel_y
            self.dino_vel_y -= self.gravity
            if self.dino_y <= 0.0:
                self.dino_y = 0.0
                self.dino_vel_y = 0.0
                self.is_jumping = False

        # Move Obstacle
        self.obstacle_x -= self.speed
        
        # Respawn Obstacle
        if self.obstacle_x < -50.0:
            self.obstacle_x = random.uniform(450.0, 650.0)
            self.obstacle_width = random.choice([24.0, 36.0, 48.0])
            self.speed = min(self.speed + 0.1, 18.0)
            reward += 5.0

        # Collision Box Checking
        ground_y = 120
        dino_rect = pygame.Rect(
            int(self.dino_x), 
            int(ground_y - self.dino_h - self.dino_y), 
            int(self.dino_w), 
            int(self.dino_h)
        )
        cactus_rect = pygame.Rect(
            int(self.obstacle_x), 
            int(ground_y - self.obstacle_height), 
            int(self.obstacle_width), 
            int(self.obstacle_height)
        )

        terminated = False
        if dino_rect.colliderect(cactus_rect):
            terminated = True
            reward = -50.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, self.score >= 5000, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((247, 247, 247)) # Classic Chrome Dino light background
        ground_y = 120

        # Draw Ground Line
        pygame.draw.line(self.screen, (83, 83, 83), (0, ground_y), (self.width, ground_y), 2)
        
        # Draw Animated Chrome Dino
        dino_draw_y = int(ground_y - self.dino_h - self.dino_y)
        draw_pixel_dino(self.screen, int(self.dino_x), dino_draw_y, frame=self.anim_frame, is_jumping=self.is_jumping)
        
        # Draw Pixel Cactus
        cactus_draw_y = int(ground_y - self.obstacle_height)
        draw_pixel_cactus(self.screen, int(self.obstacle_x), cactus_draw_y, int(self.obstacle_width), int(self.obstacle_height))
        
        # Score & Speed HUD
        txt = self.font.render(f"HI: {self.score:05d}  |  Speed: {self.speed:.1f}", True, (83, 83, 83))
        self.screen.blit(txt, (self.width - 220, 10))

        pygame.display.flip()
        self.clock.tick(60)

# ---------------------------------------------------------
# 2. Complete Execution Pipeline
# ---------------------------------------------------------
if __name__ == "__main__":
    eval_env = ChromeDinoEnv(render_mode="human")
    
    # --- PHASE 1: UNTRAINED DINO ---
    print("\n[PHASE 1] Showing UNTRAINED Chrome Dino...")
    print("Watch the Dino fail via random guessing (3 crashes)...\n")
    
    obs, _ = eval_env.reset()
    crashes = 0
    while crashes < 3:
        action = eval_env.action_space.sample()
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        
        if terminated or truncated:
            crashes += 1
            print(f"   -> Crash #{crashes}! Game resetting...")
            obs, _ = eval_env.reset()

    # --- PHASE 2: TRAINING ---
    print("\n[PHASE 2] Training AI Agent with Reinforcement Learning (PPO)...")
    print("Please wait ~20 seconds while the neural network learns clean jump timing...")
    
    train_env = ChromeDinoEnv(render_mode=None)
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=0, 
        learning_rate=0.0004, 
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01
    )
    
    model.learn(total_timesteps=80000)
    print("   Training Complete!")

    # --- PHASE 3: TRAINED DINO DEMO ---
    print("\n[PHASE 3] Launching TRAINED Chrome Dino AI!")
    print("The agent now executes pixel-perfect jumps with full running animations.\n")
    
    obs, _ = eval_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        
        if terminated or truncated:
            print("   -> Crashed at extreme speed! Restarting game...")
            obs, _ = eval_env.reset()