/* Action Buttons Module */
function renderActionButtons() {
  const container = document.getElementById('action-buttons');
  if (!container) return;

  const actions = [
    { id: 'send', label: 'Send', icon: '📤', color: 'primary', action: () => showModal('Send Money', 'Send money feature coming soon!') },
    { id: 'receive', label: 'Receive', icon: '📥', color: 'success', action: () => showModal('Receive Money', 'Your wallet address: 0xABC123...') },
    { id: 'topup', label: 'Top Up', icon: '💳', color: 'warning', action: () => showModal('Top Up', 'Top-up feature coming soon!') },
    { id: 'withdraw', label: 'Withdraw', icon: '🏧', color: 'danger', action: () => showModal('Withdraw', 'Withdrawal feature coming soon!') }
  ];

  let html = '<div class="row g-3">';
  actions.forEach(a => {
    html += `
      <div class="col-6 col-md-3">
        <button class="btn btn-${a.color} action-btn w-100 text-white" data-action="${a.id}">
          <span style="font-size:1.5rem;display:block;margin-bottom:8px;">${a.icon}</span>
          <span>${a.label}</span>
        </button>
      </div>`;
  });
  html += '</div>';
  container.innerHTML = html;

  container.querySelectorAll('button[data-action]').forEach(btn => {
    btn.addEventListener('click', function() {
      const actionId = this.getAttribute('data-action');
      const action = actions.find(a => a.id === actionId);
      if (action) action.action();
    });
  });
}

function showModal(title, message) {
  let modal = document.getElementById('wallet-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'wallet-modal';
    modal.className = 'modal fade';
    modal.tabIndex = -1;
    modal.innerHTML = `
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content" style="border-radius:12px;border:none;">
          <div class="modal-header" style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border-radius:12px 12px 0 0;">
            <h5 class="modal-title" id="wallet-modal-title"></h5>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
          </div>
          <div class="modal-body" id="wallet-modal-body" style="padding:30px;"></div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
          </div>
        </div>
      </div>`;
    document.body.appendChild(modal);
  }
  document.getElementById('wallet-modal-title').textContent = title;
  document.getElementById('wallet-modal-body').innerHTML = `<p style="font-size:1.1rem;">${message}</p>`;
  const bsModal = new bootstrap.Modal(modal);
  bsModal.show();
}

document.addEventListener('DOMContentLoaded', renderActionButtons);
