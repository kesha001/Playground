-- USER FOR OUR ACTUAL DB, user for auth service to access db, with password Aauth123 at localhost
CREATE USER 'auth_user'@'localhost' IDENTIFIED BY 'Aauth123';

CREATE DATABASE auth;

-- Give user 'auth_user' access to database auth
GRANT ALL PRIVILEGES ON auth.* TO 'auth_user'@'localhost';

USE auth;

CREATE TABLE user(
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

-- We will use these credentials to access the db via our auth service
-- This user will go to the DB which will have access to our auth service Gateway API
INSERT INTO user (email, password) VALUES ('danylo@email.com', 'Admin123');




