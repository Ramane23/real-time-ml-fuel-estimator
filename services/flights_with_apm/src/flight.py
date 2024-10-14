from pydantic import BaseModel
from  typing import Optional


class Flight(BaseModel):
    """
    A class to represent a flight using pydantic BaseModel.
    """

    aircraft_iata_code: str
    aircraft_icao_code: str
    aircraft_mtow_kg : Optional[float] = None
    aircraft_malw_kg : Optional[float] = None
    aircraft_engine_class : Optional[str] = "unknown"
    aircraft_num_engines : Optional[float] = None
    airline_iata_code: str
    airline_icao_code: str
    airline_name: str
    altitude: float
    flight_level: str
    arrival_airport_iata: str
    arrival_airport_icao: str
    arrival_city: str
    current_flight_time: int
    departure_airport_iata: str
    departure_airport_icao: str
    departure_city: str
    direction: float
    flight_icao_number: str
    flight_number: str
    flight_id: str
    flight_status: str
    horizontal_speed: float
    isGround: bool
    latitude: float
    longitude: float
    vertical_speed: float
    true_airspeed_ms: float
    departure_country: str
    arrival_country: str
    route: str
    departure_airport_lat : Optional[float] = "unknown"
    departure_airport_long : Optional[float] = "unknown"
    arrival_airport_lat : Optional[float] = "unknown"
    arrival_airport_long : Optional[float] = "unknown"

class FlightPerformance(BaseModel):
    """
    A class to represent a flight's performance data using pydantic BaseModel.
    Includes aircraft and engine characteristics, as well as calculated emissions.
    """

    # Aircraft details
    aircraft_iata_code: str
    aircraft_icao_code: str
    aircraft_mtow_kg: Optional[float] = None
    aircraft_malw_kg: Optional[float] = None
    aircraft_engine_class: Optional[str] = "unknown"
    aircraft_num_engines: Optional[int] = None
    airline_iata_code: str
    airline_icao_code: str
    airline_name: str

    # Flight details
    altitude: float
    flight_level: str
    arrival_airport_iata: str
    arrival_airport_icao: str
    arrival_city: str
    current_flight_time: int
    departure_airport_iata: str
    departure_airport_icao: str
    departure_city: str
    direction: float
    flight_icao_number: str
    flight_number: str
    flight_id: str
    flight_status: str
    horizontal_speed: float
    isGround: bool
    latitude: str
    longitude: str
    vertical_speed: float
    true_airspeed_ms: float
    departure_country: str
    arrival_country: str
    route: str
    departure_airport_lat: Optional[float] = "unknown"
    departure_airport_long: Optional[float] = "unknown"
    arrival_airport_lat: Optional[float] = "unknown"
    arrival_airport_long: Optional[float] = "unknown"

    # Engine characteristics
    engine_type: Optional[str] = None
    bypass_ratio: Optional[float] = None
    cruise_thrust: Optional[float] = None
    cruise_sfc: Optional[float] = None  # Specific Fuel Consumption at cruise
    ei_nox_to: Optional[float] = None
    ei_nox_co: Optional[float] = None
    ei_nox_app: Optional[float] = None
    ei_nox_idl: Optional[float] = None
    ei_co_to: Optional[float] = None
    ei_co_co: Optional[float] = None
    ei_co_app: Optional[float] = None
    ei_co_idl: Optional[float] = None

    # Emissions and fuel flow
    co2_flow: float
    h2o_flow: float
    nox_flow: float
    co_flow: float
    hc_flow: float
    soot_flow: float
    fuel_flow: float

    def to_dict(self):
        """
        Convert the FlightPerformance instance to a dictionary.
        """
        return {
            "aircraft_iata_code": self.aircraft_iata_code,
            "aircraft_icao_code": self.aircraft_icao_code,
            "aircraft_mtow_kg": self.aircraft_mtow_kg,
            "aircraft_malw_kg": self.aircraft_malw_kg,
            "aircraft_engine_class": self.aircraft_engine_class,
            "aircraft_num_engines": self.aircraft_num_engines,
            "airline_iata_code": self.airline_iata_code,
            "airline_icao_code": self.airline_icao_code,
            "airline_name": self.airline_name,
            "altitude": self.altitude,
            "flight_level": self.flight_level,
            "arrival_airport_iata": self.arrival_airport_iata,
            "arrival_airport_icao": self.arrival_airport_icao,
            "arrival_city": self.arrival_city,
            "current_flight_time": self.current_flight_time,
            "departure_airport_iata": self.departure_airport_iata,
            "departure_airport_icao": self.departure_airport_icao,
            "departure_city": self.departure_city,
            "direction": self.direction,
            "flight_icao_number": self.flight_icao_number,
            "flight_number": self.flight_number,
            "flight_id": self.flight_id,
            "flight_status": self.flight_status,
            "horizontal_speed": self.horizontal_speed,
            "isGround": self.isGround,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "vertical_speed": self.vertical_speed,
            "true_airspeed_ms": self.true_airspeed_ms,
            "departure_country": self.departure_country,
            "arrival_country": self.arrival_country,
            "route": self.route,
            "departure_airport_lat": self.departure_airport_lat,
            "arrival_airport_lat": self.arrival_airport_lat,
            "departure_airport_long": self.departure_airport_long,
            "arrival_airport_long": self.arrival_airport_long,
            
            # Engine characteristics
            "engine_type": self.engine_type,
            "bypass_ratio": self.bypass_ratio,
            "cruise_thrust": self.cruise_thrust,
            "cruise_sfc": self.cruise_sfc,
            "ei_nox_to": self.ei_nox_to,
            "ei_nox_co": self.ei_nox_co,
            "ei_nox_app": self.ei_nox_app,
            "ei_nox_idl": self.ei_nox_idl,
            "ei_co_to": self.ei_co_to,
            "ei_co_co": self.ei_co_co,
            "ei_co_app": self.ei_co_app,
            "ei_co_idl": self.ei_co_idl,
            
            # Emissions and fuel flow
            "co2_flow": self.co2_flow,
            "h2o_flow": self.h2o_flow,
            "nox_flow": self.nox_flow,
            "co_flow": self.co_flow,
            "hc_flow": self.hc_flow,
            "soot_flow": self.soot_flow,
            "fuel_flow": self.fuel_flow
        }
