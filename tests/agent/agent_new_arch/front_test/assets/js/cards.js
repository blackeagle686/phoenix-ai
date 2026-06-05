/* Cards Module */
function renderCards() {
  const container = document.getElementById('cards-list');
  if (!container) return;

  let html = '<div class="row">';
  mockData.cards.forEach(card => {
    const statusClass = card.status === 'active' ? 'bg-success' : 'bg-danger';
    const statusLabel = card.status === 'active' ? 'Active' : 'Frozen';
    html += `
      <div class="col-md-6 col-lg-4 mb-4">
        <div class="card credit-card ${card.tier}">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-start mb-3">
              <span class="badge badge-status ${statusClass}">${statusLabel}</span>
              <strong>${card.type.toUpperCase()}</strong>
            </div>
            <h5 class="mb-3">**** **** **** ${card.last4}</h5>
            <div class="d-flex justify-content-between small">
              <div>
                <div class="text-muted">CARD HOLDER</div>
                <div>${card.holder}</div>
              </div>
              <div>
                <div class="text-muted">EXPIRES</div>
                <div>${card.expiry}</div>
              </div>
              <div>
                <div class="text-muted">LIMIT</div>
                <div>$${card.limit.toLocaleString()}</div>
              </div>
            </div>
          </div>
        </div>
      </div>`;
  });
  html += '</div>';
  container.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', renderCards);
