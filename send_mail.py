import smtplib
import json
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def SendNNData(ToSend):
    """
    Sends mail with visulaztion attachments to a user
    
    Parameters:
    ___________
    ToSend - email address to send
    
    Returns:
    ________
    e: Error if error occurs
    None: If no errors happens
    """
    with open("userdata.json") as f:
        content = f.read()
    EmailData = json.loads(content)
    gmail_user = EmailData['gmail_address']
    gmail_password = EmailData['gmail_password']
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        
        message = MIMEMultipart()
        message["From"] = gmail_user
        message["To"] = ToSend
        message["Subject"] = "The Sonic AI has been trained"
        body = "The AI has been trained. Please find the graph attached."
        message.attach(MIMEText(body, "plain"))
        filenames = ["avg_fitness.svg", "speciation.svg", "winner.pkl"]
        for filename in filenames:
            with open(filename, "rb") as attachment:
            # The content type "application/octet-stream" means that a MIME attachment is a binary file
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            # Encode to base64
            encoders.encode_base64(part)

            # Add header 
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )
            message.attach(part)
        server.sendmail(gmail_user, ToSend, message.as_string()) 
    except Exception as e:
        print (e)

if __name__ == "__main__":
    SendNNData("aolubusi22@wooster.edu")
