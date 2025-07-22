## START SYSTEM ###
sudo systemctl daemon-reload
sudo systemctl enable sentenial-threat-monitor
sudo systemctl start sentenial-threat-monitor

### CHECK STATUS AND LOGS ###
sudo systemctl status sentenial-threat-monitor
sudo journalctl -u sentenial-threat-monitor -f
