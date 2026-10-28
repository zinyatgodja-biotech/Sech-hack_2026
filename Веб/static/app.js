// Тута только кнопки и обновление страницы.
// Все команды отправляются в Python (app.py).
const $ = id => document.getElementById(id);
let lastMessage = '';
let cachedPosition = 0;

async function request(path, method = 'GET', data) {
  const options = {method, headers: {'Content-Type': 'application/json'}};
  if (data !== undefined) options.body = JSON.stringify(data);
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Ошибка запроса');
  return result;
}

function log(text) {
  const time = new Date().toLocaleTimeString('ru-RU');
  $('log').textContent = `[${time}] ${text}\n` + $('log').textContent;
}

function readProgram() {
  return {
    travel: Number($('travel').value),
    speed: Number($('speed').value),
    hold: Number($('hold').value),
    cycles: Number($('cycles').value)
  };
}

function render(state) {
  cachedPosition = state.position;
  $('positionReadout').textContent = state.position.toFixed(2) + ' мм';
  $('pos').textContent = state.position.toFixed(2) + ' мм';
  $('mode').textContent = state.mode;
  $('stage').textContent = state.stage;
  $('cycle').textContent = `${state.cycle} / ${state.total_cycles}`;
  $('message').textContent = state.message;
  $('positionSlider').value = state.position;
  $('bubble').style.left = `${20 + state.position * 5}%`;
  const running = state.mode === 'Работает' || state.mode === 'Останавливается';
  $('start').disabled = running;
  $('positionSlider').disabled = running;
  document.querySelectorAll('[data-step]').forEach(btn => btn.disabled = running);
  if (state.message !== lastMessage) {
    log(state.message);
    lastMessage = state.message;
  }
}

async function refresh() {
  try { render(await request('/api/status')); }
  catch (error) { $('message').textContent = 'Нет связи с локальным сервером'; }
}

async function action(path, data) {
  try { render(await request(path, 'POST', data)); }
  catch (error) { log('Ошибка: ' + error.message); }
}

$('positionSlider').addEventListener('change', e => action('/api/move', {position: Number(e.target.value)}));
document.querySelectorAll('[data-step]').forEach(button => {
  button.addEventListener('click', () => {
    const position = Math.max(0, Math.min(10, cachedPosition + Number(button.dataset.step)));
    action('/api/move', {position});
  });
});
$('start').addEventListener('click', () => action('/api/start', readProgram()));
$('stop').addEventListener('click', () => action('/api/stop'));
$('emergency').addEventListener('click', () => action('/api/stop'));
$('save').addEventListener('click', async () => {
  try { const result = await request('/api/program', 'POST', readProgram()); log(result.message); }
  catch(error) { log('Ошибка: ' + error.message); }
});
$('load').addEventListener('click', async () => {
  try {
    const program = await request('/api/program');
    ['travel', 'speed', 'hold', 'cycles'].forEach(id => $(id).value = program[id]);
    log('Сохранённая программа загружена');
  } catch(error) { log('Ошибка: ' + error.message); }
});
refresh();
setInterval(refresh, 200);
