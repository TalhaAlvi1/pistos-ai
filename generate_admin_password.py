#!/usr/bin/env python3
"""
Generate SHA-256 hash for admin password.
Usage: python generate_admin_password.py
"""

import hashlib
import getpass


def generate_password_hash(password: str) -> str:
    """Generate SHA-256 hash of password."""
    return hashlib.sha256(password.encode()).hexdigest()


def main():
    print("=" * 60)
    print("Pistos Admin Password Hash Generator")
    print("=" * 60)
    print()
    
    # Get password from user
    while True:
        password = getpass.getpass("Enter new admin password: ")
        
        if len(password) < 8:
            print("❌ Password must be at least 8 characters long")
            continue
        
        confirm = getpass.getpass("Confirm password: ")
        
        if password != confirm:
            print("❌ Passwords do not match. Try again.")
            continue
        
        break
    
    # Generate hash
    password_hash = generate_password_hash(password)
    
    print()
    print("✓ Password hash generated successfully!")
    print()
    print("=" * 60)
    print("Add this to your .env file:")
    print("=" * 60)
    print()
    print(f"ADMIN_PASSWORD_HASH={password_hash}")
    print()
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. Copy the line above")
    print("2. Create or edit the .env file in the project root")
    print("3. Paste the line into .env")
    print("4. Restart the application")
    print()


if __name__ == "__main__":
    main()
