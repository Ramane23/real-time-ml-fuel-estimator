from pydantic_settings import BaseSettings
from typing import Optional, List
from pydantic import field_validator

# Define a class to hold the configuration settings
class Config(BaseSettings):
    aviation_edge_api_key: str
    kafka_broker_address: Optional[str] = None #this is optional because quixcloud will provide the kafka broker address
    kafka_topic_name: str
    live_or_historical: str
    days : Optional[float] = 1
    openap_supported_aircraft_types : List[str] = [
            'a19n', 'a20n', 'a21n', 'a318', 'a319', 'a320', 'a321', 'a332', 'a333',
            'a343', 'a359', 'a388', 'b37m', 'b38m', 'b39m', 'b3xm', 'b734', 'b737',
            'b738', 'b739', 'b744', 'b748', 'b752', 'b763', 'b772', 'b773', 'b77w',
            'b788', 'b789', 'c550', 'e145', 'e170', 'e190', 'e195', 'e75l', 'glf6'
        ]

    # Validate the value of the live_or_historical settings
    @field_validator("live_or_historical")
    def validate_live_or_historical(
        cls, value
    ):  # cls is the class itslef andd value is the value of the field
        if value not in {"live", "historical"}:
            raise ValueError("Invalid value for live_or_historical")
        return value


# Create an instance of the Config class
config = Config()
