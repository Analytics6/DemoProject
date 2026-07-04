import React, { useEffect, useState } from 'react';
import './styles.css';

function App() {
  const [user, setUser] = useState(null);
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('Admin@123!');
  const [error, setError] = useState('');
  const [stats, setStats] = useState({ active_users: 0, open_tickets: 0, audit_events: 0 });
  const [tickets, setTickets] = useState([]);

  const login = async (e) => {
    e.preventDefault();
    setError('');
    const response = await fetch('http://127.0.0.1:8000/auth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const data = await response.json();
    if (!response.ok) {
      setError(data.error || 'Login failed');
      return;
    }
    setUser(data.user);
    setStats(data.stats);
    setTickets(data.tickets);
  };

  useEffect(() => {
    const load = async () => {
      const response = await fetch('http://127.0.0.1:8000/health');
      if (response.ok) {
        const data = await response.json();
        setStats(data.stats || stats);
      }
    };
    load();
  }, []);

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <h1>Enterprise Retail Support</h1>
          <p>Modern support operations, AI assistance, and customer visibility in one portal.</p>
        </div>
      </header>

      {!user ? (
        <form className="card" onSubmit={login}>
          <h2>Sign in</h2>
          <label>Username</label>
          <input value={username} onChange={(e) => setUsername(e.target.value)} />
          <label>Password</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
          <button type="submit">Sign In</button>
          {error && <p className="error">{error}</p>}
          <div className="sample-creds">
            <strong>Sample credentials:</strong> admin / Admin@123!, agent / Agent@123!
          </div>
        </form>
      ) : (
        <div className="dashboard">
          <section className="card">
            <h2>Welcome, {user.full_name}</h2>
            <p>Role: {user.role}</p>
            <div className="metrics">
              <div><strong>{stats.active_users}</strong><span>Users</span></div>
              <div><strong>{stats.open_tickets}</strong><span>Open Tickets</span></div>
              <div><strong>{stats.audit_events}</strong><span>Audit Events</span></div>
            </div>
          </section>
          <section className="card">
            <h2>Recent Tickets</h2>
            <ul>
              {tickets.map((ticket) => (
                <li key={ticket.id}>{ticket.subject} — <b>{ticket.status}</b></li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
