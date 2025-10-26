let workCount = 0;
let educationCount = 0;
let projectCount = 0;

function addWork(data = {}) {
  const container = document.getElementById('workContainer');
  const div = document.createElement('div');
  div.className = 'work-entry';
  div.innerHTML = `
    <h4>Work ${workCount + 1}</h4>
    <label>Job Title: <input type="text" name="jobTitle${workCount}" value="${data.jobTitle || ''}"  ></label>
    <label>Company: <input type="text" name="company${workCount}" value="${data.company || ''}"  ></label>
    <label>Location: <input type="text" name="location${workCount}" value="${data.location || ''}"></label>
    <label>From: <input type="text" name="from${workCount}" value="${data.from || ''}" placeholder="YYYY-MM"  ></label>
    <label>To: <input type="text" name="to${workCount}" value="${data.to || ''}" placeholder="YYYY-MM"  ></label>
    <label>Role Description: <textarea name="role${workCount}">${data.role || ''}</textarea></label>
    <button class="delete-btn" type="button" onclick="this.parentElement.remove(); workCount--; ">Delete Work</button>
  `;
  container.appendChild(div);
  workCount++;
}

function addProject(data = {}) {
  const container = document.getElementById('projectContainer');
  const div = document.createElement('div');
  div.className = 'project-entry';
  div.innerHTML = `
    <h4>Project ${projectCount + 1}</h4>
    <label>Project Title: <input type="text" name="projectTitle${projectCount}" value="${data.title || ''}"  ></label>
    <label>Project Description: <textarea name="projectDescription${projectCount}"  >${data.description || ''}</textarea></label>
    <button class="delete-btn" type="button" onclick="this.parentElement.remove(); projectCount--; ">Delete Project</button>
  `;
  container.appendChild(div);
  projectCount++;
}

function addEducation(data = {}) {
  const container = document.getElementById('educationContainer');
  const div = document.createElement('div');
  div.className = 'education-entry';
  div.innerHTML = `
    <h4>Education ${educationCount + 1}</h4>
    <label>School: <input type="text" name="school${educationCount}" value="${data.school || ''}"  ></label>
    <label>Degree:
      <select name="degree${educationCount}" required>
        <option value="" selected disabled>Please select a degree</option>
        <option value="High School Diploma">High School Diploma</option>
        <option value="Associate">Associate</option>
        <option value="Bachelor">Bachelor</option>
        <option value="Master">Master</option>
        <option value="PhD">PhD</option>
        <option value="Diploma">Diploma</option>
        <option value="Certificate">Certificate</option>
        <option value="Other">Other</option>
      </select>
    </label>
    <label>Field of Study: <input type="text" name="field${educationCount}" value="${data.field || ''}"  ></label>
    <button class="delete-btn" type="button" onclick="this.parentElement.remove(); educationCount--; ">Delete Education</button>
  `;
  container.appendChild(div);
  educationCount++;
}

function isValidDateYYYYMM(value) {
  return /^\d{4}-(0[1-9]|1[0-2])$/.test(value);
}

document.getElementById('experienceForm').addEventListener('submit', function (e) {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(this).entries());

  const works = [];
  const educations = [];
  const projects = [];

  // Collect work experiences
  for (let i = 0; i < workCount; i++) {
    const jobTitle = data[`jobTitle${i}`];
    const company = data[`company${i}`];
    const from = data[`from${i}`];
    const to = data[`to${i}`];

    // Validate each experience entry
    if (jobTitle || company || from || to) {
      if (!jobTitle || !company || !from || !to) {
        alert(`Please complete all fields for Experience ${i + 1} (Job Title, Company, From, To).`);
        return;
      }

      if (!isValidDateYYYYMM(from) || !isValidDateYYYYMM(to)) {
        alert(`Please enter valid dates in YYYY-MM format for work ${i + 1}. Example: 2021-09`);
        return;
      }
      works.push({
        jobTitle,
        company,
        location: data[`location${i}`],
        from,
        to,
        role: data[`role${i}`]
      });
    }
  }

  // Collect educations
  // Validate and collect education entries
  for (let i = 0; i < educationCount; i++) {
    const school = data[`school${i}`]?.trim();
    const degree = data[`degree${i}`];
    const field = data[`field${i}`]?.trim();

    // If any field has data, validate the entire set
    if (school || degree || field) {
      if (!school) {
        alert(`Please enter a school name for Education ${i + 1}.`);
        return;
      }

      if (!degree) {
        alert(`Please select a degree for Education ${i + 1}.`);
        return;
      }

      if (!field) {
        alert(`Please enter a field of study for Education ${i + 1}.`);
        return;
      }

      // Everything valid → store it
      educations.push({
        school,
        degree,
        field,
      });
    }
  }


  // Collect projects
  for (let i = 0; i < projectCount; i++) {
    const title = data[`projectTitle${i}`];
    const description = data[`projectDescription${i}`];

    if (title || description) {
      if (!title || !description) {
        alert(`Please complete all fields for Project ${i + 1} (Title, Description).`);
        return;
      }
      projects.push({ title, description });
    }
  }

  // Make sure at least one section is filled
  if (works.length === 0 && educations.length === 0 && projects.length === 0) {
    alert('Please add at least one Experience, Education, or Project before continuing.');
    return;
  }

  // Validate LinkedIn if provided
  const linkedin = data.linkedin?.trim();
  if (linkedin && !/^https?:\/\/(www\.)?linkedin\.com\/.*$/i.test(linkedin)) {
    alert('Please enter a valid LinkedIn URL (starting with https://linkedin.com/).');
    return;
  }

  // Save to sessionStorage (original behavior)
  const experienceData = {
    works,
    educations,
    projects,
    skills: data.skills || '',
    linkedin: linkedin || ''
  };

  sessionStorage.setItem('experienceInfo', JSON.stringify(experienceData));
  window.location.href = 'questions.html';
});

window.addEventListener('DOMContentLoaded', () => {
  const stored = JSON.parse(sessionStorage.getItem('experienceInfo') || '{}');

  if (stored.works && stored.works.length > 0) {
    stored.works.forEach(addWork);
  } 
  // else {
  //   addWork();
  // }

  if (stored.educations && stored.educations.length > 0) {
    stored.educations.forEach(addEducation);
  }

  if (stored.projects && stored.projects.length > 0) {
    stored.projects.forEach(addProject);
  } 

  document.querySelector('[name="skills"]').value = stored.skills || '';
  document.querySelector('[name="linkedin"]').value = stored.linkedin || '';
});
