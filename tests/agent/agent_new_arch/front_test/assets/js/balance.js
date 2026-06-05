/* Balance Summary Module */
function renderBalanceSummary() {
  const section = document.getElementById('balance-summary');
  const { total, income, expenses, currency } = mockData.balance;

  section.innerHTML = `
    <div class="col-md-4 mb-3">
      <div class="card balance-card total">
        <div class="card-body text-center">
          <h6 class="card-title">Total Balance</h6>
          <p class="card-text">${currency} ${total.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p>
        </div>
      </div>
    </div>
    <div class="col-md-4 mb-3">
      <div class="card balance-card income">
        <div class="card-body text-center">
          <h6 class="card-title">Monthly Income</h6>
          <p class="card-text">${currency} ${income.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p>
        </div>
      </div>
    </div>
    <div class="col-md-4 mb-3">
      <div class="card balance-card expenses">
        <div class="card-body text-center">
          <h6 class="card-title">Monthly Expenses</h6>
          <p class="card-text">${currency} ${expenses.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p>
        </div>
      </div>
    </div>
  `;
}

document.addEventListener('DOMContentLoaded', renderBalanceSummary);
