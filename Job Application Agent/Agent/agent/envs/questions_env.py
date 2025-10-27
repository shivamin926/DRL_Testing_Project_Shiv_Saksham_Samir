from gymnasium import Env, spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import time
from agent.handler.data_loader import ApplicantManager, Questions


class QuestionsEnv(Env):

    def __init__(self, driver=None, applicant=None):
        super().__init__()

        # ---------- Chrome setup ----------
        if driver:
            self.driver = driver
            self.shared_driver = True
        else:
            self.shared_driver = False
            options = Options()
            options.add_argument("--headless=new") # hides the browser window
            options.add_argument("--disable-gpu")
            options.add_argument("--log-level=3")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), 
                options=options
            )

        self.url = "http://localhost:8000/pages/questions.html"

        # ---------- Gym setup ----------
        # 0–4: fill dropdowns, 5: submit
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # ---------- State ----------
        self.current_step = 0
        self.fields = [
            "age18",
            "canWork",
            "gender",
            "veteranStatus",
            "disabilityStatus",
        ]

        if applicant is not None:
            self.current_applicant = applicant
            self.manager = None
        else:
            self.manager = ApplicantManager(Questions)
            self.manager.load_from_json("agent/train_data/page3.json")
            self.current_applicant = None


        self.questions_filled = 0
        self.total_questions = 5


    # ---------------- Reset ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self.shared_driver:
            self.driver.get(self.url)
        else:
            if "questions.html" not in self.driver.current_url:
                self.driver.get(self.url)

        self.current_step = 0
        self.questions_filled = 0
        if self.current_applicant is None and self.manager:
            self.current_applicant = self.manager.next_applicant()

        # Reset dropdowns
        for field in self.fields:
            try:
                select = Select(self.driver.find_element(By.NAME, field))
                select.select_by_index(0)
            except Exception:
                pass

        obs = np.zeros(3, dtype=np.float32)
        info = {"page": "questions.html"}
        return obs, info

    # ---------------- Step ----------------
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {"page": "questions.html"}

        try:
            applicant = self.current_applicant

            # -------- Actions --------
            if action == 0:
                select = Select(self.driver.find_element(By.NAME, self.fields[0]))
                idx = select.options.index(select.first_selected_option)
                if idx == 0:
                    self._choose(self.fields[0], applicant.age18)
                    reward += 1
                else:
                    reward -= 1

            elif action == 1:
                select = Select(self.driver.find_element(By.NAME, self.fields[1]))
                idx = select.options.index(select.first_selected_option)
                if idx == 0:
                    self._choose(self.fields[1], applicant.can_work)
                    reward += 1
                else:
                    reward -= 1

            elif action == 2:
                select = Select(self.driver.find_element(By.NAME, self.fields[2]))
                idx = select.options.index(select.first_selected_option)
                if idx == 0:
                    self._choose(self.fields[2], applicant.gender)
                    reward += 1
                else:
                    reward -= 1

            elif action == 3:
                select = Select(self.driver.find_element(By.NAME, self.fields[3]))
                idx = select.options.index(select.first_selected_option)
                if idx == 0:
                    self._choose(self.fields[3], applicant.veteran_status)
                    reward += 1
                else:
                    reward -= 1

            elif action == 4:
                select = Select(self.driver.find_element(By.NAME, self.fields[4]))
                idx = select.options.index(select.first_selected_option)
                if idx == 0:
                    self._choose(self.fields[4], applicant.disability_status)
                    reward += 1
                else:
                    reward -= 1

            elif action == 5:  # Submit
                self._click_submit()


            # if action in range(5):  # fill one of 5 dropdowns
            #     field = self.fields[action]
            #     self._choose(field, "Yes")
            #     self.questions_filled += 1
            #     reward += 0.5
            #     if self.questions_filled == self.total_questions:
            #         reward += 2

            # elif action == 5:  # Submit
            #     self._click_submit()

            # -------- Alert check --------
            if self._has_alert():
                reward -= 1
                info["alert"] = True

            # -------- Page transition --------
            if "review.html" in self.driver.current_url:
                terminated = True
                reward += 30
                info["page"] = "----------review.html"

        except Exception as e:
            print("❌ Error:", e)
            info["error"] = str(e)
            reward -= 1

        # -------- Steps --------
        self.current_step += 1
        if self.current_step > 40:
            truncated = True

        reward -= 0.01
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    # -------- Observations --------
    def _get_observation(self):
        progress = self.questions_filled / self.total_questions
        return np.array([progress, self.questions_filled, self.total_questions], dtype=np.float32)

    # ---------------- Utilities ----------------
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
