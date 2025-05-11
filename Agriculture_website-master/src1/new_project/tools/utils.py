import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_notification(to_email, stage):
    """Sends an email to the farmer if the wheat growth stage is 'Ripening'."""
    
    sender_email = "emna.masmoudi.08@gmail.com"  # Replace with your email address
    sender_password = "hoho oxoj ieji vsao"  # Enable 'App Password' if needed

    subject = "🔔 Alert: Wheat Growth Stage Reached 'Ripening'"
    body = f"Hello,\n\nThe wheat has reached the '{stage}' stage. It is time to proceed with harvesting! 🌾\n\nBest regards,\nYour Agricultural Assistant."

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
