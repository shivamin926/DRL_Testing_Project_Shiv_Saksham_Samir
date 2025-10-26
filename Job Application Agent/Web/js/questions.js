document.getElementById('questionsForm').addEventListener('submit', function(e) {
  e.preventDefault();
  const formData = Object.fromEntries(new FormData(this).entries());

  // Validation for required questions
  if (!formData.age18) {
    alert('Please select whether you are 18 or older.');
    return;
  }
  if (!formData.canWork) {
    alert('Please select if you are legally entitled to work in Canada.');
    return;
  }
  if (!formData.gender) {
    alert('Please select an option.');
    return;
  }
  if (!formData.veteranStatus) {
    alert('Please select an option.');
    return;
  }
  if (!formData.disabilityStatus) {
    alert('Please select an option.');
    return;
  }

  // All good — save and move to next page
  sessionStorage.setItem('questionInfo', JSON.stringify(formData));
  window.location.href = 'review.html';
});

// Prefill previously saved values
window.addEventListener('DOMContentLoaded', () => {
  const data = JSON.parse(sessionStorage.getItem('questionInfo') || '{}');
  for (let key in data) {
    const element = document.querySelector(`[name="${key}"]`);
    if (element) element.value = data[key];
  }
});
