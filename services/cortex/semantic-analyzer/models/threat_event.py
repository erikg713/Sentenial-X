from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class ThreatEvent(BaseModel):
    """
    Represents a threat event for semantic analysis by RetaliationBot.
    Fields support blockchain-related threats (e.g., malicious contract calls, high-value transfers).
    """
    id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event occurrence time (UTC)")
    severity: str = Field(..., description="Threat severity: LOW, MEDIUM, HIGH, CRITICAL")
    source: str = Field(..., description="Source of the threat (e.g., wallet address, contract)")
    target: Optional[str] = Field(None, description="Target of the threat (e.g., victim address)")
    event_type: str = Field(..., description="Type of event (e.g., 'UNUSUAL_TRANSFER', 'CONTRACT_EXPLOIT')")
    chain_id: Optional[int] = Field(None, description="Blockchain chain ID (e.g., 137 for Polygon)")
    transaction_hash: Optional[str] = Field(None, description="Related transaction hash")
    details: dict = Field(default_factory=dict, description="Additional metadata (e.g., amount, method)")

    @validator("severity")
    def validate_severity(cls, value: str) -> str:
        """Ensure severity is one of the allowed values."""
        allowed = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if value.upper() not in allowed:
            raise ValueError(f"Severity must be one of {allowed}")
        return value.upper()

    @validator("transaction_hash")
    def validate_tx_hash(cls, value: Optional[str]) -> Optional[str]:
        """Validate transaction hash format (0x-prefixed, 66 chars)."""
        if value and not (value.startswith("0x") and len(value) == 66):
            raise ValueError("Invalid transaction hash")
        return value

    @validator("chain_id")
    def validate_chain_id(cls, value: Optional[int]) -> Optional[int]:
        """Ensure chain_id is valid (e.g., Polygon mainnet or Amoy)."""
        if value and value not in {137, 80002}:  # Add more supported chains as needed
            raise ValueError("Unsupported chain ID")
        return value

    class Config:
        json_encoders = {
            UUID: str,  # Serialize UUID as string
            datetime: lambda v: v.isoformat(),  # ISO 8601 for timestamps
        }
