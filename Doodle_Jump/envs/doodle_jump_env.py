"""
Doodle Jump-like Gymnasium environment using Pygame.
Coins, flying enemies, upward pellets (shoot), personas from YAML, and denser reward shaping.
"""
import os
# Headless by default for training; visualize.py unsets this for display
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import random
import pygame
import numpy as np
import yaml
from gymnasium import Env, spaces

# -------------------- Config load --------------------
def _resolve_personas_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "configs", "personas.yaml"))
    return candidate

def _load_personas():
    path = _resolve_personas_path()
    with open(path, "r") as f:
        return yaml.safe_load(f)

PERSONAS = _load_personas()

# -------------------- Game Constants --------------------
SCREEN_W = 400
SCREEN_H = 600

GRAVITY = 0.42
MOVE_ACCEL = 1.05
FRICTION = 0.82
JUMP_VELOCITY = -11.2

# Curriculum / ease
PLATFORM_W_BASE = 120
PLATFORM_H = 12
PLAT_GAP_MIN_BASE = 36
PLAT_GAP_MAX_BASE = 72
MAX_PLATFORMS = 14
INITIAL_PLATFORMS = 7
PLATFORM_HORIZONTAL_VAR = 0.55

TIME_LIMIT = 3000  # steps

# Coins / Enemies / Pellets
MAX_COINS = 6
COIN_SIZE = 12
COIN_SPAWN_P_BASE = 0.25
COIN_VERTICAL_OFFSET = 28

MAX_ENEMIES = 2
ENEMY_W, ENEMY_H = 28, 22
ENEMY_SPEED = 1.3
ENEMY_SPAWN_P_BASE = 0.14
ENEMY_MIN_HEIGHT = 160  # visible sooner

PELLET_W, PELLET_H = 6, 12
PELLET_SPEED = -10.0
PELLET_COOLDOWN = 40

# Rewards (defaults overridden by persona)
REWARD_LAND = 0.6
REWARD_CONSECUTIVE = 0.6
REWARD_CLIMB_SCALE = 1.5
REWARD_CLIMB_CAP = 6.0
REWARD_COIN = 6.0
REWARD_KILL = 2.0
PENALTY_DEATH = -6.0
PENALTY_IDLE = -0.12
PENALTY_PLATFORM_TIME = -0.03
REWARD_HEIGHT_BONUS = 0.005
REWARD_UPWARD_MOTION = 0.02
REWARD_HORIZONTAL_ACTIVITY = 0.003

COL_BG = (20, 20, 28)
COL_PLAT = (60, 200, 120)
COL_PLAYER = (240, 230, 80)
COL_COIN = (255, 200, 0)
COL_ENEMY = (220, 70, 70)
COL_PELLET = (200, 220, 255)

# -------------------- Entities --------------------
class Platform:
    __slots__ = ("x","y","w","h","pid")
    _NEXT_ID = 0
    def __init__(self, x, y, w=PLATFORM_W_BASE, h=PLATFORM_H):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.pid = Platform._NEXT_ID
        Platform._NEXT_ID += 1
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

class Player:
    __slots__ = ("x","y","vx","vy","w","h","cooldown")
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.w, self.h = 26, 32
        self.cooldown = 0
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

class Coin:
    __slots__ = ("x","y","r")
    def __init__(self, x, y, r=COIN_SIZE):
        self.x, self.y, self.r = x, y, r
    def rect(self):
        return pygame.Rect(int(self.x - self.r), int(self.y - self.r), 2*self.r, 2*self.r)

class Enemy:
    __slots__ = ("x","y","w","h","vx")
    def __init__(self, x, y, w=ENEMY_W, h=ENEMY_H, vx=ENEMY_SPEED):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.vx = vx if random.random() < 0.5 else -vx
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

class Pellet:
    __slots__ = ("x","y","w","h","vy")
    def __init__(self, x, y, w=PELLET_W, h=PELLET_H, vy=PELLET_SPEED):
        self.x, self.y, self.w, self.h, self.vy = x, y, w, h, vy
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

# -------------------- Env --------------------
class DoodleJumpEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, seed=None, reward_preset="survivor"):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.preset_name = reward_preset if reward_preset in PERSONAS else "survivor"
        self._apply_persona(PERSONAS[self.preset_name])

        # 0=left, 1=right, 2=idle, 3=shoot
        self.action_space = spaces.Discrete(4)

        # 13D observation (added onPlatform flag)
        high = np.ones((13,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._seed(seed)
        self._reset_game_state()

    # ------------- Gym API -------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
        self._reset_game_state()
        if self.render_mode == "human":
            self._init_pygame()

        # detect if starting exactly on a platform
        prect = self.player.rect()
        on_plat = any(
            abs((self.player.y + self.player.h) - p.y) <= 2 and prect.colliderect(p.rect())
            for p in self.platforms
        )
        return self._get_obs(on_plat), {"max_height": self.max_height, "persona": self.preset_name}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        self.steps += 1
        reward = 0.0

        # Tiny activity bonus to prevent freezing
        reward += REWARD_HORIZONTAL_ACTIVITY * (abs(self.player.vx) > 0.2)

        # --- Action handling ---
        if action == 0:      # left
            self.player.vx -= MOVE_ACCEL
        elif action == 1:    # right
            self.player.vx += MOVE_ACCEL
        elif action == 2:    # idle
            reward += PENALTY_IDLE
        elif action == 3:    # shoot
            if self.player.cooldown <= 0:
                px = self.player.x + self.player.w//2 - PELLET_W//2
                py = self.player.y - PELLET_H
                self.pellets.append(Pellet(px, py))
                self.player.cooldown = PELLET_COOLDOWN

        # Physics
        self.player.vx *= FRICTION
        self.player.vx = max(-6.0, min(6.0, self.player.vx))
        self.player.vy += GRAVITY

        # Cooldown decrement
        if self.player.cooldown > 0:
            self.player.cooldown -= 1

        # Track platform "camping"
        self.platform_time = getattr(self, 'platform_time', 0)
        on_platform_now = False

        # Apply motion
        self.player.x += self.player.vx
        self.player.y += self.player.vy

        # Wrap horizontally
        if self.player.x < -self.player.w:
            self.player.x = SCREEN_W
        elif self.player.x > SCREEN_W:
            self.player.x = -self.player.w

        # --- Land on platforms ---
        landed = False
        if self.player.vy > 0:  # descending
            prect = self.player.rect()
            for plat in self.platforms:
                if prect.colliderect(plat.rect()):
                    if (self.player.y + self.player.h - self.player.vy) <= plat.y + 4:
                        self.player.y = plat.y - self.player.h
                        self.player.vy = JUMP_VELOCITY
                        landed = True
                        on_platform_now = True
                        # landing counters + anti-camping logic
                        self.landings += 1
                        if self.last_platform_pid is None or plat.pid != self.last_platform_pid:
                            reward += REWARD_LAND
                            if plat.pid not in self.visited_platforms:
                                reward += 0.2  # novelty once per unique platform
                                self.visited_platforms.add(plat.pid)
                        else:
                            reward -= 0.05  # same platform again
                        self.last_platform_pid = plat.pid
                        break
        else:
            # detect "standing" on platform top (edge case)
            prect = self.player.rect()
            for plat in self.platforms:
                if abs((self.player.y + self.player.h) - plat.y) <= 2 and prect.colliderect(plat.rect()):
                    on_platform_now = True
                    break

        # On-platform time (escalating penalty + leaving bonus)
        if on_platform_now and abs(self.player.vy) < 0.1:
            self.platform_time += 1
            reward += -0.02 * min(self.platform_time, 150)
        else:
            if getattr(self, "platform_time", 0) >= 20:
                reward += 0.1  # tiny bonus for finally leaving a camp
            self.platform_time = 0

        # --- Pellets & enemies ---
        new_pellets = []
        pellet_kills = 0
        for pe in self.pellets:
            pe.y += pe.vy
            if pe.y + pe.h < 0:
                continue
            hit_idx = None
            for ei, en in enumerate(self.enemies):
                if pe.rect().colliderect(en.rect()):
                    hit_idx = ei
                    break
            if hit_idx is not None:
                pellet_kills += 1
                self.enemies.pop(hit_idx)
            else:
                new_pellets.append(pe)
        self.pellets = new_pellets
        if pellet_kills > 0:
            reward += REWARD_KILL * pellet_kills

        terminated = False
        for en in self.enemies:
            en.x += en.vx
            if en.x < -en.w:
                en.x = SCREEN_W
            elif en.x > SCREEN_W:
                en.x = -en.w
            if self.player.rect().colliderect(en.rect()):
                reward += PENALTY_DEATH
                terminated = True
                break

        # --- Camera scroll / honest climb reward ---
        if self.player.vy < -0.1:
            reward += REWARD_UPWARD_MOTION

        if self.player.y < SCREEN_H * 0.4:
            dy = SCREEN_H * 0.4 - self.player.y
            self.player.y += dy
            self._scroll(dy)

        new_max = min(self.max_height, self.global_camera_y)
        if new_max < self.max_height:
            delta = (self.max_height - new_max)
            gain = min(delta * REWARD_CLIMB_SCALE, REWARD_CLIMB_CAP)
            reward += gain
            reward += REWARD_HEIGHT_BONUS
            self.max_height = new_max

        # --- Coin collection ---
        new_coins = []
        coins_got = 0
        prect = self.player.rect()
        for c in self.coins:
            if prect.colliderect(c.rect()):
                coins_got += 1
            else:
                new_coins.append(c)
        self.coins = new_coins
        if coins_got > 0:
            reward += REWARD_COIN * coins_got

        # Maintain world & spawn
        self._ensure_platforms_and_objects()

        # --- Death by falling ---
        if not terminated and self.player.y > SCREEN_H:
            reward += PENALTY_DEATH
            terminated = True

        truncated = (self.steps >= TIME_LIMIT)

        obs = self._get_obs(on_platform_now)
        info = {
            "max_height": self.max_height,
            "steps": self.steps,
            "persona": self.preset_name,
            "death": int(terminated),
            "platforms": self.landings,  # <- used by eval
        }
        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, info

    # ------------- Render API -------------
    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                self._init_pygame()
            self._render_frame()
        elif self.render_mode == "rgb_array":
            if self.screen is None:
                self._init_pygame(headless=True)
            surface = self._draw_surface()
            return pygame.surfarray.array3d(surface).swapaxes(0,1)
        else:
            return None

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    # ------------- Internal Helpers -------------
    def _apply_persona(self, p):
        global REWARD_LAND, REWARD_CLIMB_SCALE, REWARD_CLIMB_CAP, REWARD_COIN, REWARD_KILL
        global PENALTY_DEATH, PENALTY_IDLE, PENALTY_PLATFORM_TIME, REWARD_HEIGHT_BONUS
        global REWARD_UPWARD_MOTION, COIN_SPAWN_P_BASE, ENEMY_SPAWN_P_BASE
        global PLATFORM_W_BASE, PLAT_GAP_MIN_BASE, PLAT_GAP_MAX_BASE

        REWARD_LAND = p["REWARD_LAND"]
        REWARD_CLIMB_SCALE = p["REWARD_CLIMB_SCALE"]
        REWARD_CLIMB_CAP = p["REWARD_CLIMB_CAP"]
        REWARD_COIN = p["REWARD_COIN"]
        REWARD_KILL = p["REWARD_KILL"]
        PENALTY_DEATH = p["PENALTY_DEATH"]
        PENALTY_IDLE = p["PENALTY_IDLE"]
        PENALTY_PLATFORM_TIME = p["PENALTY_PLATFORM_TIME"]
        REWARD_HEIGHT_BONUS = p["REWARD_HEIGHT_BONUS"]
        REWARD_UPWARD_MOTION = p["REWARD_UPWARD_MOTION"]
        COIN_SPAWN_P_BASE = p["COIN_SPAWN_P"]
        ENEMY_SPAWN_P_BASE = p["ENEMY_SPAWN_P"]
        PLATFORM_W_BASE = p["PLATFORM_W"]
        PLAT_GAP_MIN_BASE = p["GAP_MIN"]
        PLAT_GAP_MAX_BASE = p["GAP_MAX"]

    def _seed(self, seed):
        if seed is None:
            seed = random.randint(0, 10_000_000)
        self._rnd = random.Random(seed)
        np.random.seed(seed)

    def _reset_game_state(self):
        self.player = Player(SCREEN_W//2 - 13, SCREEN_H - 120)
        self.platforms = []
        self.coins = []
        self.enemies = []
        self.pellets = []

        self.global_camera_y = SCREEN_H  # decreases as we go up
        self.max_height = self.global_camera_y
        self.steps = 0

        # landing / camping trackers
        self.landings = 0
        self.last_platform_pid = None
        self.visited_platforms = set()
        self.platform_time = 0

        # Seed ground stack
        y = SCREEN_H - 20
        for _ in range(INITIAL_PLATFORMS):
            x = self._rnd.randint(0, SCREEN_W - PLATFORM_W_BASE)
            self.platforms.append(Platform(x, y, w=PLATFORM_W_BASE))
            if self._rnd.random() < 0.5:
                self._maybe_spawn_coin_near(x, y)
            y -= self._rnd.randint(PLAT_GAP_MIN_BASE, PLAT_GAP_MAX_BASE)

        # Safe platform under player
        safe_y = self.player.y + self.player.h + 6
        center_x = max(0, min(SCREEN_W - PLATFORM_W_BASE, int(self.player.x + self.player.w/2 - PLATFORM_W_BASE/2)))
        self.platforms.append(Platform(center_x, int(safe_y), w=PLATFORM_W_BASE))

    def _maybe_spawn_coin_near(self, px, py):
        if len(self.coins) >= MAX_COINS:
            return
        if self._rnd.random() < COIN_SPAWN_P_BASE:
            cx = int(px + PLATFORM_W_BASE//2 + self._rnd.randint(-PLATFORM_W_BASE//3, PLATFORM_W_BASE//3))
            cy = int(py - COIN_VERTICAL_OFFSET)
            self.coins.append(Coin(cx, cy))

    def _maybe_spawn_enemy_near(self, py):
        if len(self.enemies) >= MAX_ENEMIES:
            return
        # earlier visibility
        if self.global_camera_y > SCREEN_H - (ENEMY_MIN_HEIGHT // 2):
            return
        if self._rnd.random() < ENEMY_SPAWN_P_BASE:
            ex = self._rnd.randint(0, SCREEN_W - ENEMY_W)
            ey = int(py - self._rnd.randint(30, 90))
            self.enemies.append(Enemy(ex, ey))

    def _ensure_platforms_and_objects(self):
        while len(self.platforms) < MAX_PLATFORMS:
            top_y = min(p.y for p in self.platforms) if self.platforms else SCREEN_H
            new_y = top_y - self._rnd.randint(PLAT_GAP_MIN_BASE, PLAT_GAP_MAX_BASE)

            prev_x = self.platforms[-1].x if self.platforms else SCREEN_W/2
            max_x_diff = int(SCREEN_W * PLATFORM_HORIZONTAL_VAR)
            x_offset = self._rnd.randint(-max_x_diff, max_x_diff)
            x = max(0, min(SCREEN_W - PLATFORM_W_BASE, prev_x + x_offset))

            self.platforms.append(Platform(x, new_y, w=PLATFORM_W_BASE))
            self._maybe_spawn_coin_near(x, new_y)
            self._maybe_spawn_enemy_near(new_y)

        # Cull off-screen objects
        self.platforms = [p for p in self.platforms if p.y < SCREEN_H + 40]
        self.coins = [c for c in self.coins if c.y < SCREEN_H + 40]
        self.enemies = [e for e in self.enemies if e.y < SCREEN_H + 40]
        self.pellets = [pe for pe in self.pellets if pe.y + pe.h > -20]

    def _scroll(self, dy):
        for p in self.platforms:
            p.y += dy
        for c in self.coins:
            c.y += dy
        for e in self.enemies:
            e.y += dy
        for pe in self.pellets:
            pe.y += dy
        self.global_camera_y -= dy

    def _get_obs(self, on_platform: bool = False):
        px_center = self.player.x + self.player.w / 2
        py_top = self.player.y

        plats = sorted(self.platforms, key=lambda p: abs(p.y - py_top))[:2]
        coin = min(self.coins, key=lambda c: abs(c.y - py_top)) if self.coins else None
        enemy = min(self.enemies, key=lambda e: abs(e.y - py_top)) if self.enemies else None

        vals = []
        # player (4)
        vals.append((px_center - SCREEN_W/2) / (SCREEN_W/2))
        vals.append((py_top - SCREEN_H/2) / (SCREEN_H/2))
        vals.append(max(-1.0, min(1.0, self.player.vx/6.0)))
        vals.append(max(-1.0, min(1.0, self.player.vy/12.0)))

        # 2 plats (4)
        for p in plats:
            cx = p.x + p.w / 2
            cy = p.y
            relx = (cx - px_center) / (SCREEN_W/2)
            rely = (cy - py_top) / (SCREEN_H/2)
            vals.append(float(max(-1.0, min(1.0, relx))))
            vals.append(float(max(-1.0, min(1.0, rely))))
        while len(vals) < 8:
            vals.extend([0.0, 1.0])  # pad

        # coin (2)
        if coin:
            relx = ((coin.x) - px_center) / (SCREEN_W/2)
            rely = (coin.y - py_top) / (SCREEN_H/2)
            vals.append(float(max(-1.0, min(1.0, relx))))
            vals.append(float(max(-1.0, min(1.0, rely))))
        else:
            vals.extend([0.0, 1.0])

        # enemy (2)
        if enemy:
            ex_center = enemy.x + enemy.w/2
            relx = (ex_center - px_center) / (SCREEN_W/2)
            rely = (enemy.y - py_top) / (SCREEN_H/2)
            vals.append(float(max(-1.0, min(1.0, relx))))
            vals.append(float(max(-1.0, min(1.0, rely))))
        else:
            vals.extend([0.0, 1.0])

        # on-platform (1)
        vals.append(1.0 if on_platform else -1.0)

        return np.array(vals[:13], dtype=np.float32)

    def _init_pygame(self, headless=False):
        if not pygame.get_init():
            pygame.init()
        if self.screen is None:
            flags = 0
            if headless or os.environ.get("SDL_VIDEODRIVER") == "dummy":
                flags |= pygame.HIDDEN
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
            pygame.display.set_caption(f"DoodleJumpEnv ({self.preset_name})")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _draw_surface(self):
        surface = pygame.Surface((SCREEN_W, SCREEN_H))
        surface.fill(COL_BG)
        for p in self.platforms:
            pygame.draw.rect(surface, COL_PLAT, p.rect(), border_radius=4)
        for c in self.coins:
            pygame.draw.circle(surface, COL_COIN, (int(c.x), int(c.y)), c.r)
        for e in self.enemies:
            pygame.draw.rect(surface, COL_ENEMY, e.rect(), border_radius=4)
        for pe in self.pellets:
            pygame.draw.rect(surface, COL_PELLET, pe.rect(), border_radius=2)
        pygame.draw.rect(surface, COL_PLAYER, self.player.rect(), border_radius=6)
        return surface

    def _render_frame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        frame = self._draw_surface()
        self.screen.blit(frame, (0,0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
