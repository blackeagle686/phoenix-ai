/* Transactions Module */
function renderTransactions() {
  const recentContainer = document.getElementById('recent-transactions');
  const fullContainer = document.getElementById('full-transaction-list');

  if (!recentContainer && !fullContainer) return;

  const transactionsHtml = (list, limit) => {
    const items = limit ? list.slice(0, limit) : list;
    let html = '<div class="list-group">';
    items.forEach(tx => {
      const amountClass = tx.type === 'income' ? 'income' : 'expense';
      const sign = tx.type === 'income' ? '+' : '';
      html += `
        <div class="list-group-item transaction-item d-flex align-items-center">
          <div class="tx-icon ${amountClass} me-3">${tx.icon}</div>
          <div class="flex-grow-1">
            <strong>${tx.description}</strong>
            <div class="text-muted small">${tx.date} &middot; ${tx.category}</div>
          </div>
          <div class="tx-amount ${amountClass}">${sign}${tx.amount.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
        </div>`;
    });
    html += '</div>';
    return html;
  };

  if (recentContainer) {
    recentContainer.innerHTML = `
      <h5 class="mb-3">Recent Transactions</h5>
      ${transactionsHtml(mockData.transactions, 5)}
    `;
  }

  if (fullContainer) {
    fullContainer.innerHTML = `
      <div class="d-flex justify-content-between align-items-center mb-3">
        <h5 class="mb-0">All Transactions</h5>
        <input type="text" class="form-control search-bar" style="max-width: 250px;" placeholder="Search transactions..." id="tx-search">
      </div>
      ${transactionsHtml(mockData.transactions)}
    `;

    const searchInput = document.getElementById('tx-search');
    if (searchInput) {
      searchInput.addEventListener('input', function() {
        const query = this.value.toLowerCase();
        const filtered = mockData.transactions.filter(tx =>
          tx.description.toLowerCase().includes(query) ||
          tx.category.toLowerCase().includes(query)
        );
        fullContainer.innerHTML = `
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h5 class="mb-0">All Transactions</h5>
            <input type="text" class="form-control search-bar" style="max-width: 250px;" placeholder="Search transactions..." id="tx-search" value="${query}">
          </div>
          ${transactionsHtml(filtered)}
        `;
      });
    }
  }
}

document.addEventListener('DOMContentLoaded', renderTransactions);
