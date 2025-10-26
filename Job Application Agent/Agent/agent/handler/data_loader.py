import json

class Personal:
    """Represents a Personal information of a single applicant."""

    def __init__(self, data: dict):
        self.first_name = data.get("firstName", "")
        self.last_name = data.get("lastName", "")
        self.email = data.get("email", "")
        self.phone = data.get("phone", "")
        self.country = data.get("country", "")

    def as_dict(self):
        return {
            "firstName": self.first_name,
            "lastName": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "country": self.country
        }

class Experience:
    """
    Represents a single applicant's professional experience page.
    Supports multiple works, projects, and education entries,
    plus skills and LinkedIn URL.
    """
    def __init__(self, data: dict):
        # ---- Lists of sub-sections ----
        self.works = data.get("works", [])
        self.projects = data.get("projects", [])
        self.educations = data.get("education", [])

        # ---- Simple fields ----
        self.skills = data.get("skills", "")
        self.linkedin = data.get("linkedin", "")

    def as_dict(self):
        return {
            "works": self.works,
            "projects": self.projects,
            "education": self.educations,
            "skills": self.skills,
            "linkedin": self.linkedin
        }

class Questions:
    """
    Represents the applicant's responses to the demographic / eligibility questions page.
    """
    def __init__(self, data: dict):
        self.age18 = data.get("age18", "")
        self.can_work = data.get("canWork", "")
        self.gender = data.get("gender", "")
        self.veteran_status = data.get("veteranStatus", "")
        self.disability_status = data.get("disabilityStatus", "")

    def as_dict(self):
        return {
            "age18": self.age18,
            "canWork": self.can_work,
            "gender": self.gender,
            "veteranStatus": self.veteran_status,
            "disabilityStatus": self.disability_status
        }
    
class Applicant:
    """Represents a full job applicant as a single object"""

    def __init__(self, data: dict):
        # ----- Personal -----
        personal = data.get("personal", {})
        self.first_name = personal.get("firstName", "")
        self.last_name = personal.get("lastName", "")
        self.email = personal.get("email", "")
        self.phone = personal.get("phone", "")
        self.country = personal.get("country", "")

        # ----- Experience -----
        experience = data.get("experience", {})
        self.works = experience.get("works", [])
        self.projects = experience.get("projects", [])
        self.educations = experience.get("educations", experience.get("education", []))  
        self.skills = experience.get("skills", "")
        self.linkedin = experience.get("linkedin", "")

        # ----- Questions -----
        questions = data.get("questions", {})
        self.age18 = questions.get("age18", "")
        self.can_work = questions.get("canWork", "")
        self.gender = questions.get("gender", "")
        self.veteran_status = questions.get("veteranStatus", "")
        self.disability_status = questions.get("disabilityStatus", "")

    def as_dict(self):
        return {
            "personal": {
                "firstName": self.first_name,
                "lastName": self.last_name,
                "email": self.email,
                "phone": self.phone,
                "country": self.country,
            },
            "experience": {
                "works": self.works,
                "projects": self.projects,
                "educations": self.educations,
                "skills": self.skills,
                "linkedin": self.linkedin,
            },
            "questions": {
                "age18": self.age18,
                "canWork": self.can_work,
                "gender": self.gender,
                "veteranStatus": self.veteran_status,
                "disabilityStatus": self.disability_status,
            },
        }

class ApplicantManager:
    def __init__(self, class_):
        self.class_ = class_
        self.entries = []  
        self.index = 0

    def load_from_json(self, file_path: str, key="correct"):
        with open(file_path, "r") as f:
            data = json.load(f)

        records = data.get(key, [])
        if not records:
            raise ValueError(f"No entries found under key '{key}' in {file_path}")

        self.entries = [self.class_(entry) for entry in records]
        self.index = 0
        print(f"Loaded {len(self.entries)} {self.class_.__name__} entries from {file_path}")

    def next_applicant(self):
        if not self.entries:
            raise ValueError("No data loaded. Call load_from_json() first.")

        entry = self.entries[self.index]
        self.index = (self.index + 1) % len(self.entries)
        print(f"Serving {self.class_.__name__} #{self.index}: {entry}")
        return entry

    def reset_cycle(self):
        """Reset index to start from first entry again."""
        self.index = 0
        print("Data cycle reset.")
