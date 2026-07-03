CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    loyalty_tier TEXT DEFAULT 'standard',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users (full_name, email, loyalty_tier) VALUES
('Aarav Sharma', 'aarav.sharma@example.com', 'gold'),
('Priya Nair', 'priya.nair@example.com', 'silver'),
('Rohan Verma', 'rohan.verma@example.com', 'standard');
