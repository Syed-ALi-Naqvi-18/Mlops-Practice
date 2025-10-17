from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to HousePK - API Feature Added"

@app.route('/login')
def login():
    return "Login page by Ali"

if __name__ == "__main__":
    app.run(debug=True)
