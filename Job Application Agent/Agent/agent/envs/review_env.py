from gymnasium import Env, spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import time
import json


class ReviewEnv(Env):

    def __init__(self, driver=None, applicant=None):
        super().__init__()

        # ---------- Chrome setup ----------
        if driver:
            self.driver = driver
            self.shared_driver = True
        else:
            self.shared_driver = False
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--log-level=3")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options,
            )

        self.url = "http://localhost:8000/pages/review.html"

        # ---------- Gym setup ----------
        self.action_space = spaces.Discrete(1)     # 0 = click submit
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # ---------- State ----------
        self.current_step = 0
        self.submitted = False
        self.alert_closed = False

    # ---------------- Reset ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self.shared_driver:
            self.driver.get(self.url)
        else:
            if "review.html" not in self.driver.current_url:
                self.driver.get(self.url)

        self.current_step = 0
        self.submitted = False
        self.alert_closed = False

        obs = np.zeros(2, dtype=np.float32)
        info = {"page": "review.html"}
        return obs, info

    # ---------------- Step ----------------
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {"page": "review.html"}

        try:
            if action == 0 and not self.submitted:
                self._click_submit()
                self.submitted = True
                reward += 1  

                # Wait for potential alert
                time.sleep(0.5)

                if self._has_alert():
                    if self._alert_success_check():
                        reward += 5
                        self.alert_closed = True
                    else:
                        reward -= 1

                # ---------- Validate review data ----------
                result, ratio = self._validate_review_data()
                info["result"] = result
                info["match_ratio"] = ratio

                # Reward shaping
                if result == "done":
                    reward += 10
                elif result == "partial":
                    reward += 3
                else:
                    reward -= 3

                # Complete episode
                reward += 5
                terminated = True
                info["page"] = "done"

            elif self.submitted:
                reward -= 1

        except Exception as e:
            print("❌ Error:", e)
            info["error"] = str(e)
            reward -= 1

        obs = np.zeros(2, dtype=np.float32)
        self.current_step += 1
        if self.current_step > 10:
            truncated = True

        reward -= 0.01
        return obs, reward, terminated, truncated, info

    # ---------------- Helper Functions ----------------
    def _click_submit(self):
        try:
            button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            button.click()
            time.sleep(0.3)
        except Exception as e:
            print("⚠️ Submit click failed:", e)

    def _has_alert(self):
        try:
            alert = self.driver.switch_to.alert
            print(f"⚠️ Alert detected: {alert.text}")
            alert.accept()
            return True
        except NoAlertPresentException:
            return False

    def _alert_success_check(self):
        try:
            alert = self.driver.switch_to.alert
            if "application submitted successfully" in alert.text.lower():
                alert.accept()
                return True
            else:
                return False
        except NoAlertPresentException:
            return False

    # ---------- Review validation ----------
    def _validate_review_data(self):
        """Compare review page data with the applicant object."""
        try:
            script = """
            return JSON.stringify({
                personal: JSON.parse(sessionStorage.getItem('personalInfo') || '{}'),
                experience: JSON.parse(sessionStorage.getItem('experienceInfo') || '{}'),
                questions: JSON.parse(sessionStorage.getItem('questionInfo') || '{}')
            });
            """
            data_str = self.driver.execute_script(script)
            review_data = json.loads(data_str)
            applicant_dict = self.applicant.as_dict() if self.applicant else {}

            def compare(d1, d2):
                matches, total = 0, 0
                for k, v in d1.items():
                    if isinstance(v, dict) and k in d2:
                        m, t = compare(v, d2[k])
                        matches += m
                        total += t
                    else:
                        total += 1
                        if str(d2.get(k, "")).strip() == str(v).strip():
                            matches += 1
                return matches, total

            matches, total = compare(applicant_dict, review_data)
            ratio = matches / total if total else 0
            result = "done" if ratio > 0.95 else "partial" if ratio > 0.5 else "mismatch"
            print(f" Review match: {matches}/{total} ({ratio:.1%}) → {result.upper()}")
            return result, ratio
        except Exception as e:
            print(f"⚠️ Review validation failed: {e}")
            return "error", 0.0

    # ---------- Close ----------
    def close(self):
        if not self.shared_driver:
            try:
                self.driver.quit()
                print("Browser closed.")
            except Exception as e:
                print(f"Close failed: {e}")
