// Clock & greeting
function updateClock() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2, '0');
  const m = String(now.getMinutes()).padStart(2, '0');
  const s = String(now.getSeconds()).padStart(2, '0');
  document.getElementById('clock').textContent = `${h}:${m}:${s}`;

  const days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  document.getElementById('date').textContent =
    `${days[now.getDay()]}, ${months[now.getMonth()]} ${now.getDate()}`;

  const hour = now.getHours();
  let greeting = 'Good morning';
  if (hour >= 12 && hour < 17) greeting = 'Good afternoon';
  else if (hour >= 17) greeting = 'Good evening';
  document.getElementById('greeting').textContent = `${greeting}. Here's your overview.`;
}

updateClock();
setInterval(updateClock, 1000);

// Nav
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    item.classList.add('active');
  });
});

// Tasks
document.getElementById('addTaskBtn').addEventListener('click', () => {
  const form = document.getElementById('addTaskForm');
  form.classList.toggle('hidden');
  if (!form.classList.contains('hidden')) {
    document.getElementById('newTaskInput').focus();
  }
});

document.getElementById('newTaskInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') addTask();
  if (e.key === 'Escape') {
    document.getElementById('addTaskForm').classList.add('hidden');
  }
});

function addTask() {
  const input = document.getElementById('newTaskInput');
  const text = input.value.trim();
  if (!text) return;

  const li = document.createElement('li');
  li.className = 'task-item';
  li.innerHTML = `
    <span class="task-check" onclick="toggleTask(this)"></span>
    <span class="task-text">${text}</span>
  `;
  document.getElementById('taskList').appendChild(li);

  input.value = '';
  document.getElementById('addTaskForm').classList.add('hidden');
}

function toggleTask(el) {
  const item = el.closest('.task-item');
  item.classList.toggle('done');
  el.textContent = item.classList.contains('done') ? '✓' : '';
}
