# plugins/notify_on_detections.py

import smtplib
from email.message import EmailMessage
from sentenial_core.adapters.db_adapter import DBAdapter

def register(register_command):
    register_command("notify_admin", notify_admin)

def notify_admin(payload: str) -> str:
    """
    Send an email summary to the admin.
    Expects 'email,subject,body' as comma-separated args.
    """
    try:
        email, subject, body = [p.strip() for p in payload.split(",", 2)]
    except ValueError:
        return "Usage: notify_admin <email>,<subject>,<body>"

    msg = EmailMessage()
    msg["To"] = email
    msg["Subject"] = subject
    msg.set_content(body)

    # NOTE: configure your SMTP settings here!
    with smtplib.SMTP("localhost") as smtp:
        smtp.send_message(msg)

    # Log the notification in memory
    DBAdapter().log_memory({"action": "notify", "to": email, "subject": subject})
    return f"Notification sent to {email}."
