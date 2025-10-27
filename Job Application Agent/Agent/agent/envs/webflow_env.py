import gymnasium as gym
from gymnasium import spaces
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from agent.envs.index_env import IndexEnv
from agent.envs.experience_env import ExperienceEnv
from agent.envs.questions_env import QuestionsEnv
from agent.envs.review_env import ReviewEnv
from agent.handler.data_loader import ApplicantManager, Applicant


class WebFlowEnv(gym.Env):
    """Unified Gym environment that navigates through all 4 application pages sequentially."""

<<<<<<< Updated upstream
    def __init__(self):
=======
    def __init__(self, applicant=None):
>>>>>>> Stashed changes
        super().__init__()

        # ---------- Shared Chrome driver ----------
        chrome_options = Options()
        # chrome_options.add_argument("--headless=new")  # hides the browser window
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--log-level=3")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        print("Shared Chrome driver started")

        # ---------- Dataset ----------
        self.manager = ApplicantManager(Applicant)
        self.manager.load_from_json("agent/train_data/full_applicants.json", key="applicants")
<<<<<<< Updated upstream
        self.current_applicant = None
=======
        self.current_applicant = applicant
>>>>>>> Stashed changes

        # ---------- Page flow ----------
        self.pages = [IndexEnv, ExperienceEnv, QuestionsEnv, ReviewEnv]
        self.current_env = None
        self.page_index = 0

        # ---------- Gym setup ----------
        # Fixed obs shape = 8 main features + 4 one-hot page indicator
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    # ============================================================
    # Utility: pad observation and append one-hot page indicator
    # ============================================================
    def _pad_obs(self, obs):
        obs = np.array(obs, dtype=np.float32).flatten()
        if obs.shape[0] < 8:
            obs = np.pad(obs, (0, 8 - obs.shape[0]), mode="constant")
        elif obs.shape[0] > 8:
            obs = obs[:8]

        page_vec = np.zeros(4, dtype=np.float32)
        page_vec[self.page_index] = 1.0
        return np.concatenate([obs, page_vec])

    # ---------------- Reset ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Pick next applicant
<<<<<<< Updated upstream
        self.current_applicant = self.manager.next_applicant()
=======
        if self.current_applicant is None:
            self.current_applicant = self.manager.next_applicant()
>>>>>>> Stashed changes
        self.page_index = 0

        # Start first page with the shared driver
        self.current_env = self.pages[self.page_index](
            driver=self.driver, applicant=self.current_applicant
        )
        print(f"Starting new applicant flow on page {self.page_index}: index.html")

        obs, info = self.current_env.reset()
        obs = self._pad_obs(obs)
        return obs, info

    # ---------------- Step ----------------
    def step(self, action):
        # Step through the current sub-environment
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        obs = self._pad_obs(obs)

        if terminated:
            page_name = info.get("page", "")
            print(f"Page completed: {page_name}")

            # ---------- End of full application ----------
            if "done" in page_name:
                result = info.get("result", "incomplete")
                match_ratio = info.get("match_ratio", 0.0)

                # ---------- Final reward shaping ----------
                # map match_ratio (0–1) to big bonuses or penalties
                if match_ratio < 0.10:
                    final_bonus = -100
                elif match_ratio < 0.50:
                    final_bonus = 0
                elif match_ratio < 0.70:
                    final_bonus = 50
                elif match_ratio < 0.80:
                    final_bonus = 150
                else: 
                    final_bonus = 300

                reward += final_bonus

                print(f"Episode summary → Result: {result.upper()}, "
                    f"Match: {match_ratio:.1%}, Bonus: {final_bonus:+}")

                final_info = {
                    "page": "done",
                    "result": result,
                    "match_ratio": match_ratio,
                }

                return obs, reward, True, truncated, final_info

            # ---------- Early termination (error or alert) ----------
            elif self.page_index >= len(self.pages) - 1:
                final_info = {
                    "page": info.get("page", "unknown"),
                    "result": "incomplete",
                    "match_ratio": 0.0,
                }
                print("⚠️ Episode ended early (incomplete applicant)")
                return obs, reward, True, truncated, final_info

            # ---------- Move to next page ----------
            else:
                self.page_index += 1
                next_env_cls = self.pages[self.page_index]
                print(f"Moving to next page ({self.page_index})")

                self.current_env = next_env_cls(
                    driver=self.driver, applicant=self.current_applicant
                )
                obs, info = self.current_env.reset()
                obs = self._pad_obs(obs)
                terminated = False

        info.setdefault("page", f"page_{self.page_index}")
        info.setdefault("result", "incomplete")
        info.setdefault("match_ratio", 0.0)

        return obs, reward, terminated, truncated, info

    # ---------- Close ----------
    def close(self):
        if self.driver:
            try:
                self.driver.quit()
                print("Shared browser closed.")
            except Exception as e:
                print(f"Close failed: {e}")
