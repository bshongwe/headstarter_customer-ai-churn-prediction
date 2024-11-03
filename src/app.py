from flask import Flask, jsonify, request
from database import session
from models import Customer

app = Flask(__name__)

@app.route('/customers', methods=['POST'])
def create_customer():
    data = request.json
    new_customer = Customer(
        name=data['name'],
        age=data['age'],
        account_balance=data['account_balance'],
        activity_history=data['activity_history']
    )
    session.add(new_customer)
    session.commit()
    return jsonify({'message': 'Customer created!'}), 201

if __name__ == "__main__":
    app.run(debug=True)
