import bcrypt

passwords = ['Kishan1985@']

hashed_passwords = [bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') for password in passwords]
print(hashed_passwords)
