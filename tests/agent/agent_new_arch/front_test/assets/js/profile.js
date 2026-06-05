/* Profile Module */
function renderProfile() {
  const container = document.getElementById('profile-info');
  if (!container) return;

  const user = mockData.user;
  const initial = user.name.split(' ').map(n => n[0]).join('').toUpperCase();
  const memberSince = new Date(user.memberSince).toLocaleDateString('en-US', {
    year: 'numeric', month: 'long', day: 'numeric'
  });

  container.innerHTML = `
    <div class="profile-section text-center">
      <div class="profile-avatar">${initial}</div>
      <h4>${user.name}</h4>
      <p class="text-muted mb-4">${user.email}</p>
      <div class="row text-start">
        <div class="col-md-6 mb-3">
          <label class="text-muted small">Phone</label>
          <p class="mb-0"><strong>${user.phone}</strong></p>
        </div>
        <div class="col-md-6 mb-3">
          <label class="text-muted small">Member Since</label>
          <p class="mb-0"><strong>${memberSince}</strong></p>
        </div>
        <div class="col-md-6 mb-3">
          <label class="text-muted small">User ID</label>
          <p class="mb-0"><strong>${user.id}</strong></p>
        </div>
        <div class="col-md-6 mb-3">
          <label class="text-muted small">Default Currency</label>
          <p class="mb-0"><strong>${mockData.balance.currency}</strong></p>
        </div>
      </div>
      <hr>
      <h6 class="text-start mb-3">Account Summary</h6>
      <div class="row text-start">
        <div class="col-4 text-center">
          <p class="mb-0" style="font-size:1.5rem;font-weight:700;color:#667eea;">${mockData.transactions.length}</p>
          <small class="text-muted">Transactions</small>
        </div>
        <div class="col-4 text-center">
          <p class="mb-0" style="font-size:1.5rem;font-weight:700;color:#667eea;">${mockData.cards.length}</p>
          <small class="text-muted">Cards</small>
        </div>
        <div class="col-4 text-center">
          <p class="mb-0" style="font-size:1.5rem;font-weight:700;color:#28a745;">${mockData.cards.filter(c => c.status === 'active').length}</p>
          <small class="text-muted">Active Cards</small>
        </div>
      </div>
    </div>
  `;
}

document.addEventListener('DOMContentLoaded', renderProfile);
