/* Footer Module */
function renderFooter() {
  const footer = document.getElementById('main-footer');
  if (!footer) return;

  footer.innerHTML = `
    <div class="container-fluid">
      <div class="row align-items-center">
        <div class="col-md-4 text-md-start text-center mb-2 mb-md-0">
          <small class="text-muted">&copy; 2024 MyWallet. All rights reserved.</small>
        </div>
        <div class="col-md-4 text-center mb-2 mb-md-0">
          <small class="text-muted">Secure &middot; Encrypted &middot; Trusted</small>
        </div>
        <div class="col-md-4 text-md-end text-center">
          <small class="text-muted">v1.0.0</small>
        </div>
      </div>
    </div>
  `;
}

document.addEventListener('DOMContentLoaded', renderFooter);
