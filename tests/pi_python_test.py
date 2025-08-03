pi_python_test.py

import unittest from unittest.mock import patch from pi_python import PiNetwork

class TestPiNetwork(unittest.TestCase): def setUp(self): self.pi = PiNetwork() self.pi.initialize("test_api_key", "test_seed", "Pi Testnet") self.payment_data = { "amount": 3.14, "memo": "Test - Greetings from MyApp", "metadata": {"product_id": "apple-pie-1"}, "uid": "test-user" }

@patch("pi_python.requests.post")
def test_create_payment(self, mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"identifier": "test_payment_id"}

    payment_id = self.pi.create_payment(self.payment_data)
    self.assertEqual(payment_id, "test_payment_id")

@patch("pi_python.requests.post")
def test_submit_payment(self, mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"txid": "test_txid"}

    txid = self.pi.submit_payment("test_payment_id")
    self.assertEqual(txid, "test_txid")

@patch("pi_python.requests.post")
def test_complete_payment(self, mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"status": {"developer_completed": True}}

    result = self.pi.complete_payment("test_payment_id", "test_txid")
    self.assertTrue(result["status"]["developer_completed"])

@patch("pi_python.requests.get")
def test_get_payment(self, mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"identifier": "test_payment_id"}

    payment = self.pi.get_payment("test_payment_id")
    self.assertEqual(payment["identifier"], "test_payment_id")

@patch("pi_python.requests.post")
def test_cancel_payment(self, mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"status": "cancelled"}

    response = self.pi.cancel_payment("test_payment_id")
    self.assertEqual(response["status"], "cancelled")

@patch("pi_python.requests.get")
def test_get_incomplete_server_payments(self, mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = []

    payments = self.pi.get_incomplete_server_payments()
    self.assertEqual(payments, [])

if name == 'main': unittest.main()

