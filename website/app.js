const hero = document.querySelector('#hero-video');
const toggle = document.querySelector('#film-toggle');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let userPausedHero = false;
function syncPlayButton() {
  const playing = !hero.paused;
  toggle.setAttribute('aria-label', playing ? 'Pause demonstration' : 'Play demonstration');
  toggle.querySelector('span').textContent = playing ? 'Pause' : 'Play';
  toggle.querySelector('path').setAttribute('d', playing ? 'M6 5h4v14H6zm8 0h4v14h-4z' : 'm8 5 11 7-11 7z');
}
toggle.addEventListener('click', () => {
  if (hero.paused) { userPausedHero = false; hero.play().catch(syncPlayButton); }
  else { userPausedHero = true; hero.pause(); }
});
hero.addEventListener('play', syncPlayButton);
hero.addEventListener('pause', syncPlayButton);
if (!reduceMotion.matches) hero.play().catch(syncPlayButton);
new IntersectionObserver(entries => {
  if (!entries[0].isIntersecting) hero.pause();
  else if (!reduceMotion.matches && !userPausedHero && !document.hidden) hero.play().catch(syncPlayButton);
}, {threshold: 0.2}).observe(hero);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) document.querySelectorAll('video').forEach(video => video.pause());
});
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(selected, focus = false) {
  tabs.forEach(tab => {
    const active = selected === tab;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    const panel = document.getElementById(tab.getAttribute('aria-controls'));
    panel.hidden = !active;
    if (!active) panel.querySelector('video').pause();
  });
  if (focus) selected.focus();
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', event => {
    const next = {ArrowRight:(index+1)%tabs.length, ArrowLeft:(index+tabs.length-1)%tabs.length, Home:0, End:tabs.length-1}[event.key];
    if (next !== undefined) { event.preventDefault(); selectTab(tabs[next], true); }
  });
});
document.querySelectorAll('.demo-panel video').forEach(video => {
  video.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(other => { if (other !== video) other.pause(); });
  });
});
document.querySelectorAll('[data-copy]').forEach(button => {
  const original = button.textContent;
  button.addEventListener('click', async () => {
    const text = document.getElementById(button.dataset.copy).textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied';
      document.querySelector('#copy-status').textContent = `${original === 'Copy BibTeX' ? 'BibTeX' : 'Commands'} copied to clipboard.`;
      setTimeout(() => { button.textContent = original; }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(document.getElementById(button.dataset.copy));
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      button.textContent = 'Select and copy';
      document.querySelector('#copy-status').textContent = 'Text selected. Use your keyboard to copy.';
    }
  });
});
