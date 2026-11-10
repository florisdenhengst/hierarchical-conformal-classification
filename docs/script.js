// Copy-to-clipboard for the BibTeX block.
document.querySelectorAll('button[data-copy]').forEach(function (button) {
  var source = document.getElementById(button.dataset.copy);
  if (!source || !navigator.clipboard) return;

  var label = button.textContent;
  button.addEventListener('click', function () {
    navigator.clipboard.writeText(source.textContent).then(function () {
      button.textContent = 'Copied';
      setTimeout(function () { button.textContent = label; }, 1500);
    });
  });
});
