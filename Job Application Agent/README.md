# DRL for Job Application Website

## Overview
This project applies Deep Reinforcement Learning (DRL) to navigate and submit a multi-page web application that simulates a real-world job application process.  
Each page in the flow (Personal Info → Experience → Questions → Review) is implemented as an independent Gymnasium environment.  
This allows an RL agent to learn valid navigation and form-filling behavior while automatically detecting validation issues and incomplete submissions.

Two DRL algorithms, **PPO** and **A2C**, are used to train autonomous agents that can navigate through all pages, complete entries, and submit the final application successfully.

---

## Application Flow 

| Page | HTML | JS Controller | Environment Class | Description |
|------|------|----------------|-------------------|--------------|
| Personal Info | `index.html` | `personal.js` | `IndexEnv` | Handles name, email, phone, and country validation. |
| Experience | `experience.html` | `experience.js` | `ExperienceEnv` | Adds/removes dynamic work, project, and education entries. |
| Questions | `questions.html` | `questions.js` | `QuestionsEnv` | Demographic and eligibility fields with dropdowns. |
| Review | `review.html` | `review.js` | `ReviewEnv` | Validates and submits all collected form data. |

Each page has its own reward design, observations, and termination conditions.  
The `WebFlowEnv` class environment chains these sub-environments for full end-to-end agent execution.

---

## Environment Design
**NOTE: Rewards and steps for all the individual environments have been changed during various times during training and development. All the related metrics are based on latest info**
### IndexEnv (Personal Info)
- **Observations:** 5 binary features indicating which fields are filled.
- **Actions:** Fill individual fields or submit.
- **Reward structure:**
  - +2 per valid field filled.
  - +60 for successfully transitioning to the next page.
  - −1 for triggering validation alerts.
  - −0.01 time penalty per step.

### ExperienceEnv
- Handles dynamic work, project, and education sections.
- **Reward structure:**
  - +0.5 for correctly filled entry.
  - +1 bonus for completing a section.
  - -1 for trying to fill extra fields
  - +35 for full submission; 
  - −5 for partial submission.
  - −0.01 time penalty.

### QuestionsEnv
- Five dropdown-based questions for eligibility and demographics.
- **Reward structure:**
  - +1 for each correct dropdown selection.
  - −1 for revisiting already answered questions.
  - +30 for reaching the review page.
  - −1 for alerts.
  - −0.01 time penalty.

### ReviewEnv
- Compares and validates all collected information with the applicant’s dataset.
- **Reward structure:**
  - +1 for successful submission action.
  - +5 for valid submission alert.
  - +10 for a complete review match, 
  - +3 for partial review match, 
  - −3 for review mismatch.
  - −0.01 time penalty.
- The environment calculates a final match ratio between the filled data and the reference applicant profile.

### WebFlowEnv
- Combines all environments into a unified browser session for full sequential training.
- Match ratio is calculated using how much of the data from applicant data was fully utilized over all the pages.
- **Final reward bonus:**
  | Match Ratio | Reward Bonus |
  |--------------|---------------|
  | < 0.10 | −100 |
  | 0.10–0.50 | 0 |
  | 0.50–0.70 | +50 |
  | 0.70–0.80 | +150 |
  | ≥ 0.80 | +300 |

---

## Algorithms and Training

### PPO
- Framework: Stable-Baselines3  
- Environment wrapped with `Monitor` for structured logging.
- Parameters:
  ```python
    verbose=1,
    n_steps=256,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
  ```
- Trained with checkpoints every 5,000 steps and continued training to 100,000 timesteps.

### A2C
- Trained using the same framework and environments.
- Achieved comparable success rate after tuning.
- Parameters:
  ```python
    verbose=1,
    learning_rate=7e-4,
    n_steps=512,
    gamma=0.99,
    ent_coef=0.05,
    vf_coef=0.25,
    max_grad_norm=0.5,
  ```

---

## Training Summary

| Page | Timesteps | n_steps | Actions | Reward Adjustments | Outcome |
|------|------------|---------|----------|--------------------|----------|
| index | 2000 | 256 | 20 | Increased transition reward, reduced penalty | Terminated early |
| experience | 50000 | 256 | 50 | Added section rewards, time penalty | Reached questions page |
| questions | 10000 | 256 | 40 | Added revisit penalty, boosted next-page reward | Reached review page |
| review | 100 | 10 | 10 | Default rewards | Completed submission |

---

## Reproducibility

### Setup
- requirements.txt can be found at ```Analysis/requirements.txt```.
```bash
pip install -r requirements.txt
python -m http.server 8000   # serve website locally
```

### Training Commands
- Data to be used for training or testing needs to be JSON formatted. Refer to following directory for the structure ```Agent/agent/train_data/{json samples}```
- Data to be present in the same directory as mentioned above.
```bash
<<<<<<< Updated upstream
# Full sequential webflow
python train_full_ppo.py

# Single-page debug
python train_single_ppo.py
=======
#Training
# Full sequential webflow PPO
python python -m agent.src.train_full_ppo
# Full sequential webflow A2C
python python -m agent.src.train_full_a2c

# Single-page 
python python -m agent.src.train_single.ppo

#Testing
# Full sequential webflow PPO
python python -m agent.src.test_ppo
# Full sequential webflow A2C
python python -m agent.src.test_ppo
>>>>>>> Stashed changes
```

### Repository Structure
```
Agent/
  agent/
    envs/
      index_env.py
      experience_env.py
      questions_env.py
      review_env.py
      webflow_env.py
    handler/
      data_loader.py
    train_data/
      page1.json
      page2.json
      page3.json
      full_applicants.json
    src/
      train_single_ppo.py
      train_full_ppo.py
      train_full_a2c.py
  models/
    pages/
      {pages}
    a2c_full/
      {iterations}
    ppo_full/
      {iterations}
  logs/
    a2c/
      full_webflow/
        fail1.csv
        pass.csv
    ppo/
      full_webflow/
        fail1.csv
        pass.csv
      page1/
        fail1.csv
        pass.csv
      page2/
        fail1.csv
        pass.csv
      page3/
        fail1.csv
        pass.csv
      page4/
        fail1.csv
        pass.csv

Web/
  index.html
  pages/
    experience.html
    questions.html
    review.html
  js/
    personal.js
    experience.js
    questions.js
    review.js
  css/
    styles.css
Analysis/
  a2c-episode-applications/
  ppo-episode-applications/
  Images/
  Scripts/
  notebook.ipynb
  observations.csv
  requirements/txt
```

## Metrics and logging

- Logs can be found at ```Agent/logs/```.
- Tranings can be found at ```Agent/models/```.
- Please check ```Analysis/notebook.ipynb``` for comparisons and metrics.

## Demonstration Video

Below is a short clip showing the trained PPO agent filling and submitting a job application automatically.

![Training Demo](gifs/training_demo.gif)

## Summary (Tasks completed)
- Two trained DRL agents (PPO, A2C).
- Modular environments for each application page.
- Custom reward functions for issue detection and form completeness.
- End-to-end agent flow and validation.
- Exported CSV logs.
- Reproducible scripts and setup instructions.
- Separate notebook for graphs and evaluation metrics.