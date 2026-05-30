import uuid
from datetime import datetime

def generate_unique_id() -> str:
    """Generates a unique hexadecimal ID."""
    return uuid.uuid4().hex

def generate_timestamped_filename(suffix: str) -> str:
    """Generates a unique filename with a timestamp and suffix."""
    timestamp = datetime.utcnow().isoformat().replace(':', '-')
    unique_id = generate_unique_id()
    return f"{timestamp}_{unique_id}_{suffix}"

