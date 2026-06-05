/* Header Module */
function renderHeader() {
  const header = document.getElementById('main-header');
  header.innerHTML = `
    <div class="row align-items-center">
      <div class="col-6">
        <h2>💳 MyWallet</h2>
      </div>
      <div class="col-6 text-end">
        <span class="badge bg-light text-dark me-2">Welcome, ${mockData.user.name}</span>
        <button class="btn btn-outline-light btn-sm" onclick="alert('Notifications: No new alerts')">🔔</button>
      </div>
    </div>
  `;
}

document.addEventListener('DOMContentLoaded', renderHeader);
