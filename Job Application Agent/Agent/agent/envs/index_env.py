from gymnasium import Env, spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import time
from agent.handler.data_loader import Personal, ApplicantManager

class IndexEnv(Env):

    def __init__(self, driver=None, applicant=None):
        super().__init__()

        # ---------------- Chrome Setup ----------------
        if driver:
            self.driver = driver
            self.shared_driver = True
        else:
            self.shared_driver = False
            options = Options()
            options.add_argument("--headless=new")  # hides the browser window
            options.add_argument("--disable-gpu")
            options.add_argument("--log-level=3")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )

        self.url = "http://localhost:8000/index.html"

        # ---------------- Gym Setup ----------------
        # 0–5: fill fields, 6: submit
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        # ---------------- State ----------------
        self.current_step = 0
        self.fields = [
            "firstName",
            "lastName",
            "email",
            "country",
            "phone",
        ]

        if applicant is not None:
            self.current_applicant = applicant
            self.manager = None  # use provided applicant
        else:
            self.manager = ApplicantManager(Personal)
            self.manager.load_from_json("agent/train_data/page1.json")
            self.current_applicant = None

    # ---------------- Reset ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self.shared_driver:
            self.driver.get(self.url)
        else:
            if "index.html" not in self.driver.current_url:
                self.driver.get(self.url)

        self.current_step = 0
        if self.current_applicant is None and self.manager:
            self.current_applicant = self.manager.next_applicant()

        # Clear fields
        for field in self.fields:
            try:
                if field != "country":
                    select = self.driver.find_element(By.NAME, field)
                    select.clear()
            except Exception:
                pass

        # Reset country dropdown
        try:
            select = Select(self.driver.find_element(By.NAME, "country"))
            select.select_by_index(0)
        except Exception:
            pass

        obs = np.zeros(5, dtype=np.float32)
        info = {"page": "index.html"}
        return obs, info

    # ---------------- Step ----------------
    def step(self, action):
        """Take one step (agent chooses an action)."""
        reward = 0
        terminated = False
        truncated = False
        info = {"page": "index.html"}
        applicant = self.current_applicant

        try:
            # Actions
            if action == 0:
                self._fill("firstName", applicant.first_name)
                reward += 2
            elif action == 1:
                self._fill("lastName", applicant.last_name)
                reward += 2
            elif action == 2:
                self._fill("email", applicant.email)
                reward += 2
            elif action == 3:
                self._choose("country", applicant.country)
                reward += 2
            elif action == 4:
                self._fill("phone", applicant.phone)
                reward += 2
            elif action == 5:
                self._click_submit()

            # -------- Alert check --------
            if self._has_alert():
                reward -= 1
                info["alert"] = True

            # -------- Page transition --------
            if "experience.html" in self.driver.current_url:
                reward += 60
                terminated = True
                info["page"] = "experience.html"

        except Exception as e:
            print("❌ Error:", e)
            info["error"] = str(e)
            reward -= 1

        # -------- Observation --------
        obs = self._get_observation()

        # -------- Steps --------
        self.current_step += 1
        if self.current_step > 40:
            truncated = True

        # time penalty
        reward -= 0.01
        return obs, reward, terminated, truncated, info

    # -------- Observations --------
    def _get_observation(self):
        obs = []
        for field in self.fields:
            try:
                if field == "country":
                    select = Select(self.driver.find_element(By.NAME, field))
                    idx = select.options.index(select.first_selected_option)
                    obs.append(0 if idx == 0 else 1)
                else:
                    value = self.driver.find_element(By.NAME, field).get_attribute("value")
                    obs.append(0 if value.strip() == "" else 1)
            except Exception:
                obs.append(0)
        return np.array(obs, dtype=np.float32)

    # ---------------- Helper Functions ----------------
    def _fill(self, name, value):
        elem = self.driver.find_element(By.NAME, name)
        elem.clear()
        elem.send_keys(value)
        # time.sleep(0.1)

    def _choose(self, name, value):
        select = Select(self.driver.find_element(By.NAME, name))
        select.select_by_visible_text(value)
        # time.sleep(0.1)

    def _click_submit(self):
        button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        button.click()
        # time.sleep(0.5)

    def _has_alert(self):
        try:
            alert = self.driver.switch_to.alert
            print("⚠️ Alert detected:", alert.text)
            alert.accept()
            return True
        except NoAlertPresentException:
            return False

    def close(self):
        if not self.shared_driver:
            try:
                self.driver.quit()
                print("Browser closed.")
            except Exception as e:
                print(f"Close failed: {e}")