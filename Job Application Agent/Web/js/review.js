function displayReview() {
  const personal = JSON.parse(sessionStorage.getItem('personalInfo') || '{}');
  const experience = JSON.parse(sessionStorage.getItem('experienceInfo') || '{}');
  const questions = JSON.parse(sessionStorage.getItem('questionInfo') || '{}');

  let html = '<h3>Personal Information</h3>';
  for (let key in personal) {
    html += `<p><strong>${key}:</strong> ${personal[key]}</p>`;
  }

  html += '<h3>Experience</h3>';
  experience.works?.forEach((exp, i) => {
    html += `<h4>Job #${i + 1}</h4>`;
    for (let k in exp) html += `<p><strong>${k}:</strong> ${exp[k]}</p>`;
  });

  html += '<h3>Projects</h3>';
  experience.projects?.forEach((proj, i) => {
    html += `<h4>Project #${i + 1}</h4>`;
    for (let k in proj) html += `<p><strong>${k}:</strong> ${proj[k]}</p>`;
  });

  html += '<h3>Education</h3>';
  experience.educations?.forEach((edu, i) => {
    html += `<h4>School #${i + 1}</h4>`;
    for (let k in edu) html += `<p><strong>${k}:</strong> ${edu[k]}</p>`;
  });

  html += `<h3>Skills</h3><p>${experience.skills}</p>`;
  html += `<h3>LinkedIn</h3><p>${experience.linkedin}</p>`;

  html += '<h3>Application Questions</h3>';
  for (let key in questions) {
    html += `<p><strong>${key}:</strong> ${questions[key]}</p>`;
  }

  document.getElementById('reviewContent').innerHTML = html;
}

document.getElementById('submitButton').addEventListener('click', () => {
  const data = {
    personal: JSON.parse(sessionStorage.getItem('personalInfo') || '{}'),
    experience: JSON.parse(sessionStorage.getItem('experienceInfo') || '{}'),
    questions: JSON.parse(sessionStorage.getItem('questionInfo') || '{}')
  };
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'application.json';
  a.click();
  alert('Application submitted successfully!');
});

window.addEventListener('DOMContentLoaded', displayReview);
