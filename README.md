# DRL_Testing Project

This repository contains two separate projects used for coursework and experiments:

## 1) Doodle_Jump
- Purpose: Reinforcement learning experiments using a Doodle Jump-like environment.
- Location: `Doodle_Jump/`
- Important: Before running any commands for this project you MUST change directory into the project's folder and follow the README contained there. Each project folder contains its own `README.md` with full setup and run instructions.
- Quick start (from repository root):
   ```powershell
   # change into the project folder first
   Set-Location .\Doodle_Jump

   # create and activate a Python virtual environment (example)
   python -m venv .\venv
   .\venv\Scripts\Activate.ps1

   # install dependencies
   python -m pip install -r .\requirements.txt
   ```

## 2) Job Application Agent
- Purpose: DRL agent to train and test of job application webflow, with multi-page environments
- Location: `Job_Application_Agent/`.
- Important: Change into the project's folder and follow the `README.md` inside that folder before running any setup or application commands.
- Quick start (from repository root):
   ```powershell
   # change into the project folder first
   Set-Location .\Job_Application_Agent

   # create and activate a Python virtual environment (example)
   python -m venv .\venv
   .\venv\Scripts\Activate.ps1
<<<<<<< Updated upstream

   # install dependencies
   python -m pip install -r .\requirements.txt
   ```
=======
>>>>>>> Stashed changes

   # install dependencies
   python -m pip install -r .\requirements.txt
   ```
## Presentation Video
Presentation video can be found on the following link.
```
```
## General notes
- Prefer `python -m pip install` so pip runs with the same Python interpreter as your virtual environment.
- If you get "Could not open requirements file" errors, confirm your current directory with `Get-Location` (PowerShell) or `pwd` (bash), or provide the full path to the requirements file.

---
Last updated: October 2025