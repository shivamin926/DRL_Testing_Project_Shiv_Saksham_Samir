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

## 2) Job Application
- Purpose: A separate project included in this repository (internally referred to as "Job Application").
- Location: check the repository root for the `Job_Application/` (or similarly named) folder.
- Important: Change into the project's folder and follow the `README.md` inside that folder before running any setup or application commands.

If a project folder doesn't have a README yet, open it and I can create a detailed one for you.

## General notes
- Prefer `python -m pip install` so pip runs with the same Python interpreter as your virtual environment.
- If you get "Could not open requirements file" errors, confirm your current directory with `Get-Location` (PowerShell) or `pwd` (bash), or provide the full path to the requirements file.

---
Last updated: October 2025