/* Mock Wallet Data */
const mockData = {
  user: {
    id: 'usr_001',
    name: 'Alex Johnson',
    email: 'alex.johnson@example.com',
    phone: '+1 (555) 123-4567',
    memberSince: '2021-03-15',
    avatar: null
  },
  balance: {
    total: 24530.75,
    income: 8750.00,
    expenses: 3420.50,
    currency: 'USD'
  },
  transactions: [
    { id: 'tx_001', date: '2024-01-15', description: 'Salary Deposit', category: 'Income', amount: 5200.00, type: 'income', icon: '💰' },
    { id: 'tx_002', date: '2024-01-14', description: 'Grocery Store', category: 'Food', amount: -85.30, type: 'expense', icon: '🛒' },
    { id: 'tx_003', date: '2024-01-13', description: 'Electric Bill', category: 'Utilities', amount: -120.00, type: 'expense', icon: '⚡' },
    { id: 'tx_004', date: '2024-01-12', description: 'Freelance Project', category: 'Income', amount: 1500.00, type: 'income', icon: '💻' },
    { id: 'tx_005', date: '2024-01-11', description: 'Restaurant', category: 'Food', amount: -65.40, type: 'expense', icon: '🍽️' },
    { id: 'tx_006', date: '2024-01-10', description: 'Gas Station', category: 'Transport', amount: -45.00, type: 'expense', icon: '⛽' },
    { id: 'tx_007', date: '2024-01-09', description: 'Subscription Netflix', category: 'Entertainment', amount: -15.99, type: 'expense', icon: '🎬' },
    { id: 'tx_008', date: '2024-01-08', description: 'Gym Membership', category: 'Health', amount: -50.00, type: 'expense', icon: '🏋️' },
    { id: 'tx_009', date: '2024-01-07', description: 'Dividend Payment', category: 'Income', amount: 320.00, type: 'income', icon: '📈' },
    { id: 'tx_010', date: '2024-01-06', description: 'Online Shopping', category: 'Shopping', amount: -230.00, type: 'expense', icon: '🛍️' },
    { id: 'tx_011', date: '2024-01-05', description: 'Coffee Shop', category: 'Food', amount: -6.50, type: 'expense', icon: '☕' },
    { id: 'tx_012', date: '2024-01-04', description: 'Rent Payment', category: 'Housing', amount: -1800.00, type: 'expense', icon: '🏠' }
  ],
  cards: [
    { id: 'card_001', type: 'visa', last4: '4242', holder: 'Alex Johnson', expiry: '12/26', status: 'active', tier: 'platinum', limit: 10000 },
    { id: 'card_002', type: 'mastercard', last4: '8888', holder: 'Alex Johnson', expiry: '06/25', status: 'active', tier: 'premium', limit: 5000 },
    { id: 'card_003', type: 'amex', last4: '1234', holder: 'Alex Johnson', expiry: '03/27', status: 'frozen', tier: 'standard', limit: 3000 }
  ]
};
