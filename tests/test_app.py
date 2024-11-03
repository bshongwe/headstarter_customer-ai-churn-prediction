import unittest
from app import app

class CustomerTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_create_customer(self):
        response = self.app.post('/customers', json={
            'name': 'John Doe',
            'age': 30,
            'account_balance': 1500.00,
            'activity_history': 'active'
        })
        self.assertEqual(response.status_code, 201)

if __name__ == '__main__':
    unittest.main()
