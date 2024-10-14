from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional, Dict

# Define a class to hold the configuration settings
class Config(BaseSettings):
    kafka_broker_address: Optional[str] = None
    kafka_input_topic_name: str
    kafka_consumer_group: str
    kafka_output_topic_name: str
    openap_supported_aircraft_types : List[str] = [
            'a19n', 'a20n', 'a21n', 'a318', 'a319', 'a320', 'a321', 'a332', 'a333',
            'a343', 'a359', 'a388', 'b37m', 'b38m', 'b39m', 'b3xm', 'b734', 'b737',
            'b738', 'b739', 'b744', 'b748', 'b752', 'b763', 'b772', 'b773', 'b77w',
            'b788', 'b789', 'c550', 'e145', 'e170', 'e190', 'e195', 'e75l', 'glf6'
        ]
        #this is because openap fuel flow and emission models are not available for all aircraft types (it's a fallback)
    aircraft_type_mapping : Dict = {
            'a21n': 'a320',   # Map A321neo to A320
            'a20n': 'a320',   # Map A320neo to A320
            'a318': 'a320',   # Map A318 to A320
            'b38m': 'b737',   # Map B737 MAX to B737
            'e190': 'a320',   # Map Embraer E190 to Embraer E170
            'e195': 'a320',   # Map E195 to E170
            'e145': 'a320',   # Map E145 to E170
            'b734': 'b737', # Map B737-400 to B737   
            'b744': 'b737',
            'b789': 'b737',
            'b788': 'b737',
            'a359': 'a320',
            'b763': 'b737',
            'b772': 'b737',
            'E75L': 'a320',
            # Add more mappings if necessary
        }

# Create an instance of the Config class
config = Config()