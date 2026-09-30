// ISL Project Portal Client Script (Vanilla JS, Zero-Tooling)

document.addEventListener('DOMContentLoaded', () => {
  // Modal handlers
  const runLocallyBtn = document.getElementById('btn-run-locally');
  const modal = document.getElementById('run-modal');
  const closeModalBtn = document.getElementById('modal-close-btn');

  if (runLocallyBtn && modal) {
    runLocallyBtn.addEventListener('click', (e) => {
      e.preventDefault();
      modal.classList.add('active');
    });
  }

  if (closeModalBtn && modal) {
    closeModalBtn.addEventListener('click', () => {
      modal.classList.remove('active');
    });

    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.remove('active');
      }
    });
  }

  // Copy to clipboard buttons
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      const targetElement = document.getElementById(targetId);
      if (targetElement) {
        navigator.clipboard.writeText(targetElement.innerText.trim()).then(() => {
          const originalText = btn.innerText;
          btn.innerText = 'Copied!';
          setTimeout(() => {
            btn.innerText = originalText;
          }, 2000);
        });
      }
    });
  });
});
