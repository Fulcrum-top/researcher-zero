from pydantic import BaseModel


class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str