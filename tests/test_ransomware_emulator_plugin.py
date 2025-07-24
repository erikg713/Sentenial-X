import unittest
from unittest.mock import MagicMock, patch
from plugins.ransomware_emulator import RansomwareEmulatorPlugin

class TestRansomwareEmulatorPlugin(unittest.TestCase):

    @patch("plugins.ransomware_emulator.RansomwareEmulator")
    def test_parameters(self, MockEmulator):
        # Mock payloads
        mock_instance = MockEmulator.return_value
        mock_instance.list_payloads.return_value = {"fake_payload": "Fake Description"}
        
        plugin = RansomwareEmulatorPlugin()
        param_names = [param["name"] for param in plugin.parameters]

        self.assertIn("payload_name", param_names)
        self.assertIn("file_count", param_names)
        self.assertIn("monitor", param_names)

    @patch("plugins.ransomware_emulator.RansomwareEmulator")
    def test_run_campaign(self, MockEmulator):
        mock_instance = MockEmulator.return_value
        mock_instance.run_campaign.return_value = {"status": "success"}

        plugin = RansomwareEmulatorPlugin()
        result = plugin.run(payload_name="fake_payload", file_count=5, monitor=True)

        mock_instance.run_campaign.assert_called_once_with(
            payload_name="fake_payload",
            file_count=5,
            monitor=True
        )
        self.assertEqual(result["status"], "success")

if __name__ == "__main__":
    unittest.main()
