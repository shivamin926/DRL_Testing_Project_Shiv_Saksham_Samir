import gymnasium as gym
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import time
from agent.handler.data_loader import ApplicantManager, Experience

class ExperienceEnv(gym.Env):

    def __init__(self, driver=None, applicant=None):
        super().__init__()

        # ---------------- Chrome Setup ----------------
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

        self.url = "http://localhost:8000/pages/experience.html"

        # ------------------ Gym setup ------------------
        # 0–5: fill fields, 6: submit
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        # ------------------ State ------------------
        self.current_step = 0

        # Progress trackers (per episode)
        self.work_filled = 0
        self.project_filled = 0
        self.education_filled = 0
        self.skills_filled = False
        self.linkedin_filled = False

        if applicant is not None:
            self.current_applicant = applicant
            self.manager = None
        else:
            self.manager = ApplicantManager(Experience)
            self.manager.load_from_json("agent/train_data/page2.json")
            self.current_applicant = None

    # ------------------ Reset ------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.driver.get(self.url)

        self.current_step = 0
        if self.current_applicant is None and self.manager:
            self.current_applicant = self.manager.next_applicant()

        # Reset progress
        self.work_filled = 0
        self.project_filled = 0
        self.education_filled = 0
        self.skills_filled = False
        self.linkedin_filled = False

        # Clear fields
        self._clear_all_entries()
        self._clear_field("skills")
        self._clear_field("linkedin")

        obs = self._get_observation()
        info = {"page": "experience.html"}
        return obs, info

    # ------------------ Step ------------------
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {"page": "experience.html"}

        applicant = self.current_applicant
        work_len = len(applicant.works)
        proj_len = len(applicant.projects)
        edu_len = len(applicant.educations)

        self._make_fields()

        try:
            # ---------- ACTIONS ----------
            if action == 0:  # Fill next Work entry
                if self.work_filled < work_len:
                    self._fill_entry("work", self.work_filled, applicant.works[self.work_filled])
                    self.work_filled += 1
                    reward += 0.5
                    if self.work_filled == work_len:
                        reward += 2  # section completion bonus
                else:
                    reward -= 1

            elif action == 1:  # Fill next Project entry
                if self.project_filled < proj_len:
                    self._fill_entry("project", self.project_filled, applicant.projects[self.project_filled])
                    self.project_filled += 1
                    reward += 0.5
                    if self.project_filled == proj_len:
                        reward += 2
                else:
                    reward -= 1

            elif action == 2:  # Fill next Education entry
                if self.education_filled < edu_len:
                    self._fill_entry("education", self.education_filled, applicant.educations[self.education_filled])
                    self.education_filled += 1
                    reward += 0.5
                    if self.education_filled == edu_len:
                        reward += 2
                else:
                    reward -= 1

            elif action == 3:  # Fill Skills
                if not self.skills_filled:
                    self._fill("skills", applicant.skills)
                    self.skills_filled = True
                    reward += 1

            elif action == 4:  # Fill LinkedIn
                if not self.linkedin_filled:
                    self._fill("linkedin", applicant.linkedin)
                    self.linkedin_filled = True
                    reward += 1

            elif action == 5:  # Submit
                fully_filled = self._is_fully_filled(applicant)

                if fully_filled:
                    reward += 35
                    info["page"] = "questions.html"
                else:
                    reward -= 5
                    info["page"] = "questions.html--partial"

                self._click_submit()
                time.sleep(0.4)

            if self._has_alert():
                reward -= 1
                info["alert"] = True

            if "questions.html" in self.driver.current_url:
                terminated = True
                if fully_filled:
                    reward += 1
                else:
                    info["page"] = "questions.html--partial"


        except Exception as e:
            print("❌ Error:", e)
            info["error"] = str(e)
            reward -= 1

        # Time penalty
        reward -= 0.01

        # Step accounting
        self.current_step += 1
        if self.current_step > 60:
            truncated = True

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    # ------------------ Observation helper ------------------
    def _get_observation(self):
        try:
            work = len(self.driver.find_elements(By.CLASS_NAME, "work-entry"))
            proj = len(self.driver.find_elements(By.CLASS_NAME, "project-entry"))
            edu  = len(self.driver.find_elements(By.CLASS_NAME, "education-entry"))

            alert = 1.0 if self._has_alert() else 0.0
            page  = 1.0 if "questions.html" in self.driver.current_url else 0.0

            # normalize by actual applicant lengths
            work_norm = max(len(self.current_applicant.works), 1)
            proj_norm = max(len(self.current_applicant.projects), 1)
            edu_norm  = max(len(self.current_applicant.educations), 1)

            obs = np.array([
                work / work_norm, proj / proj_norm, edu / edu_norm,
                self.work_filled / work_norm,
                self.project_filled / proj_norm,
                self.education_filled / edu_norm,
                alert, page
            ], dtype=np.float32)

        except Exception as e:
            print("Observation error:", e)
            obs = np.zeros(8, dtype=np.float32)
        return obs


    # ------------------ Section creation (EXACT counts) ------------------
    def _make_fields(self):
        """Create exactly len(works/projects/educations) entries by clicking Add buttons."""

        target_work = len(self.current_applicant.works)
        target_proj = len(self.current_applicant.projects)
        target_edu  = len(self.current_applicant.educations)
        # Work
        while len(self.driver.find_elements(By.CLASS_NAME, "work-entry")) < target_work:
            self._click_button("Add Work")
        # Projects
        while len(self.driver.find_elements(By.CLASS_NAME, "project-entry")) < target_proj:
            self._click_button("Add Project")
        # Education
        while len(self.driver.find_elements(By.CLASS_NAME, "education-entry")) < target_edu:
            self._click_button("Add School")

    # ------------------ Fillers ------------------
    def _fill_entry(self, section, index, entry):
        """Fill fields for a specific section type with correct subfield names."""
        try:
            if section == "work":
                mapping = {
                    f"jobTitle{index}": entry.get("jobTitle", ""),
                    f"company{index}": entry.get("company", ""),
                    f"location{index}": entry.get("location", ""),
                    f"from{index}": entry.get("from", ""),
                    f"to{index}": entry.get("to", ""),
                    f"role{index}": entry.get("role", "")
                }
            elif section == "project":
                mapping = {
                    f"projectTitle{index}": entry.get("title", ""),
                    f"projectDescription{index}": entry.get("description", "")
                }
            elif section == "education":
                mapping = {
                    f"school{index}": entry.get("school", ""),
                    f"degree{index}": entry.get("degree", ""),
                    f"field{index}": entry.get("field", "")
                }
            else:
                mapping = {}

            for name, value in mapping.items():
                if "degree" in name:
                    self._choose(name, value)
                else:
                    self._fill(name, value)
        except Exception as e:
            print(f"Error filling {section} #{index}:", e)

    # ------------------ DOM Utilities ------------------
    def _is_fully_filled(self, applicant):
        """Return True if all sections are completely filled."""
        return (
            self.work_filled == len(applicant.works)
            and self.project_filled == len(applicant.projects)
            and self.education_filled == len(applicant.educations)
            and self.skills_filled
            and self.linkedin_filled
        )

    def _click_button(self, label_text):
        try:
            btn = self.driver.find_element(By.XPATH, f"//button[contains(text(),'{label_text}')]")
            btn.click()
            # time.sleep(0.15)
            return True
        except Exception as e:
            print(f"Could not click button '{label_text}':", e)
            return False

    def _fill(self, name, value):
        element = self.driver.find_element(By.NAME, name)
        element.clear()
        element.send_keys(value)
        # time.sleep(0.05)

    def _choose(self, name, value):
        element = self.driver.find_element(By.NAME, name)
        Select(element).select_by_visible_text(value)
        # time.sleep(0.05)

    def _click_submit(self):
        button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        button.click()
        # time.sleep(0.3)


    def _clear_all_entries(self):
        """Remove existing dynamic entries before each reset."""
        try:
            buttons = self.driver.find_elements(By.CSS_SELECTOR, "button.delete-btn")
            for btn in buttons:
                btn.click()
                # time.sleep(0.05)
        except Exception:
            pass

    def _clear_field(self, name):
        try:
            el = self.driver.find_element(By.NAME, name)
            el.clear()
        except Exception:
            pass

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

