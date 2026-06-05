/* Main Application Entry Point */
/* Initializes all modules on DOMContentLoaded */

(function() {
  'use strict';

  function init() {
    if (typeof renderHeader === 'function') renderHeader();
    if (typeof renderBalanceSummary === 'function') renderBalanceSummary();
    if (typeof renderActionButtons === 'function') renderActionButtons();
    if (typeof renderTransactions === 'function') renderTransactions();
    if (typeof renderCards === 'function') renderCards();
    if (typeof renderProfile === 'function') renderProfile();
    if (typeof renderFooter === 'function') renderFooter();
    initTabNavigation();
  }

  function initTabNavigation() {
    const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
      tab.addEventListener('shown.bs.tab', function(event) {
        const targetId = event.target.getAttribute('data-bs-target');
        if (targetId === '#transactions' && typeof renderTransactions === 'function') {
          renderTransactions();
        } else if (targetId === '#cards' && typeof renderCards === 'function') {
          renderCards();
        } else if (targetId === '#profile' && typeof renderProfile === 'function') {
          renderProfile();
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();