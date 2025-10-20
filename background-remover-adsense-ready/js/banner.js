
(function(){
  const key='br-consent';
  if(localStorage.getItem(key)) return;
  const bar = document.createElement('div');
  bar.className='banner';
  bar.innerHTML = 'We use basic cookies for analytics and to show ads. <a href="/privacy.html">Learn more</a> <button id="ok">OK</button>';
  document.body.appendChild(bar);
  bar.querySelector('#ok').addEventListener('click', ()=>{
    localStorage.setItem(key,'1'); bar.remove();
  });
})();
