import smtplib
from email.mime.text import MIMEText
import logging

class AlertDispatcher:
    def __init__(self):
        self.logger = logging.getLogger("AlertDispatcher")

        # Customize these for email alerts
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender = "your_email@gmail.com"
        self.password = "your_app_password"
        self.recipients = ["admin@example.com"]

    def send(self, message):
        self.logger.info(f"ALERT: {message}")
        try:
            msg = MIMEText(message)
            msg['Subject'] = "🚨 Threat Alert"
            msg['From'] = self.sender
            msg['To'] = ", ".join(self.recipients)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())

        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")