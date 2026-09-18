-- SQL Initialization Script for IMDB Sentiment App
-- Run this script as MySQL admin/root to set up the database, table, and user permissions.

CREATE DATABASE IF NOT EXISTS imdb_sentiment_db;

USE imdb_sentiment_db;

CREATE TABLE IF NOT EXISTS review_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    review_text LONGTEXT NOT NULL,
    predicted_label VARCHAR(20) NOT NULL,
    positive_probability FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Example: Grant least-privilege access to application user (non-root)
-- CREATE USER IF NOT EXISTS 'imdb_app_user'@'%' IDENTIFIED BY 'your_secure_password';
-- GRANT SELECT, INSERT ON imdb_sentiment_db.review_history TO 'imdb_app_user'@'%';
-- FLUSH PRIVILEGES;
