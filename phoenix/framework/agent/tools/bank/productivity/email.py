import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from phoenix.framework.agent.tools.base import BaseTool, ToolResult

class EmailTool(BaseTool):
    name = "email"
    description = (
        "Sends an email via SMTP. "
        "Inputs: 'recipient' (str, the email address to send to), "
        "'subject' (str, email subject), "
        "'body' (str, email body text)."
    )

    async def execute(
        self, 
        recipient: str, 
        subject: str, 
        body: str, 
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        try:
            # Load config from env if not explicitly provided
            server = smtp_server or os.environ.get("SMTP_SERVER", "smtp.gmail.com")
            port = smtp_port or int(os.environ.get("SMTP_PORT", 587))
            sender = sender_email or os.environ.get("SMTP_EMAIL")
            password = sender_password or os.environ.get("SMTP_PASSWORD")
            
            if not sender or not password:
                return ToolResult(
                    success=False, 
                    output=None, 
                    error="SMTP_EMAIL and SMTP_PASSWORD must be provided via arguments or environment variables."
                )

            # Build the email
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect and send
            with smtplib.SMTP(server, port) as smtp:
                smtp.starttls()
                smtp.login(sender, password)
                smtp.send_message(msg)
                
            return ToolResult(success=True, output=f"Email successfully sent to {recipient}.")
            
        except smtplib.SMTPException as e:
            return ToolResult(success=False, output=None, error=f"SMTP Error: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to send email: {str(e)}")

