document.getElementById('personalForm').addEventListener('submit', function (e) {
  e.preventDefault();

  const formData = Object.fromEntries(new FormData(this).entries());
  const firstName = formData.firstName?.trim();
  const lastName = formData.lastName?.trim();
  const email = formData.email?.trim();
  const phone = formData.phone?.trim();

  const namePattern = /^[A-Za-z\s'-]+$/;
  if (!namePattern.test(firstName) || !namePattern.test(lastName)) {
    alert('Names can only contain letters, spaces, hyphens, or apostrophes.');
    return;
  }
  if (!firstName || !lastName) {
    alert('Please enter both your first and last name.');
    return;
  }

  if (!email) {
    alert('Please enter your email address.');
    return;
  }
  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailPattern.test(email)) {
    alert('Please enter a valid email address.');
    return;
  }

  if (!phone) {
    alert('Please enter your phone number.');
    return;
  }
  const phonePattern = /^\+\d{8,15}$/;
  if (!phonePattern.test(phone)) {
    alert('Please enter a valid phone number starting with + followed by 8-15 numbers.');
    return;
  }

  if (!formData.country) {
    alert('Please select your country from the list.');
    return;
  }

  // Save and continue (original behavior)
  sessionStorage.setItem('personalInfo', JSON.stringify(formData));
  window.location.href = 'pages/experience.html';
});

// Keep previous values filled in when returning to this page
window.addEventListener('DOMContentLoaded', () => {
  const data = JSON.parse(sessionStorage.getItem('personalInfo') || '{}');
  for (let key in data) {
    const input = document.querySelector(`[name="${key}"]`);
    if (input) input.value = data[key];
  }
});
