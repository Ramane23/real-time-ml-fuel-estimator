import pandas as pd
from loguru import logger
import warnings
from typing import Tuple
from openap import prop, FuelFlow, Emission

from src.config import config
from src.flight import Flight, FlightPerformance

#ignore warnings
warnings.filterwarnings('ignore')

class AddAircraftPerformances:
    """
    Class to engineer features for the contrail predictor using OpenAP aircraft performance model.
    """

    def __init__(self):
        self.openap_supported_aircraft_types = config.openap_supported_aircraft_types
        self.aircraft_type_mapping = config.aircraft_type_mapping

    def add_aircraft_performances(
        self,
        flight: Flight
    ) -> FlightPerformance:
        """
        Adds aircraft performances to a flight object.
        
        Args:
            flight (Flight): a flight object
        
        Returns:
            FlightPerformance: a flight performance object
        """
        logger.debug('Adding aircraft performances...')
        
        #first ensure the aircraft type is in lower case
        flight.aircraft_icao_code = flight.aircraft_icao_code.lower()

        # Step 1: Add engine type
        engine_type = self.get_engine_type(flight)
        
        # Step 2: Add engine characteristics
        flight = self.add_engine_characteristics(flight, engine_type)
        
        # Step 3: Calculate fuel flow and emissions
        flight_performance = self.calculate_fuel_and_emissions(flight)

        return flight_performance

    def get_engine_type(
        self, 
        flight: Flight
        ):
        """
        Adds engine type to a flight object.
        
        Args:
            flight (Flight): a flight object
        
        Returns:
            Flight: Updated flight object with engine type added.
        """
        logger.debug('Adding engine type to the flight object...')

        try:
            aircraft_data = prop.aircraft(flight.aircraft_icao_code.upper())
            engine_type = aircraft_data['engine']['default']
            #cast the flight object to a dictionary
            #flight = flight.dict()
            #flight['engine_type'] = engine_info
            #convert the dictionary back to a Flight object
            #flight = Flight(**flight)
            #breakpoint()
            return engine_type
        except KeyError:
            logger.error(f"Engine information not found for aircraft type: {flight.aircraft_icao_code}")
            return None

    def add_engine_characteristics(
        self, 
        flight: Flight,
        engine_type: str
        ):
        """
        Adds engine characteristics to a flight object.
        
        Args:
            flight (Flight): a flight object
        
        Returns:
            Flight: Updated flight object with engine characteristics added.
        """
        logger.debug('Adding engine characteristics to the flight object...')

        #engine_type = flight.engine_type
        if engine_type and engine_type != "unknown":
            #first cast the flight object to a dictionary
            flight = flight.dict()
            characteristics = self.get_engine_characteristics(engine_type)
            flight["engine_type"] = engine_type
            flight["bypass_ratio"] = characteristics.get('bypass_ratio', None)
            flight["cruise_thrust"] = characteristics.get('cruise_thrust', None)
            flight["cruise_sfc"] = characteristics.get('cruise_sfc', None)
            flight["ei_nox_to"] = characteristics.get('ei_nox_to', None)
            flight["ei_nox_co"] = characteristics.get('ei_nox_co', None)
            flight["ei_nox_app"] = characteristics.get('ei_nox_app', None)
            flight["ei_nox_idl"] = characteristics.get('ei_nox_idl', None)
            flight["ei_co_to"] = characteristics.get('ei_co_to', None)
            flight["ei_co_co"] = characteristics.get('ei_co_co', None)
            flight["ei_co_app"] = characteristics.get('ei_co_app', None)
            flight["ei_co_idl"] = characteristics.get('ei_co_idl', None)
            #convert the dictionary back to a Flight object
            #flight = Flight(**flight)
            #breakpoint()
        else:
            logger.warning(f"Engine type not found for flight {flight.flight_id}")

        return flight

    def get_engine_characteristics(
        self, 
        engine_type: str
        ) -> dict:
        """
        Retrieves engine characteristics from OpenAP data.
        
        Args:
            engine_type (str): The engine type identifier.
        
        Returns:
            dict: A dictionary of engine characteristics.
        """
        try:
            engine = prop.engine(engine_type)
            return {
                'engine_type': engine_type,
                'bypass_ratio': engine.get('bpr', None),
                'cruise_thrust': engine.get('cruise_thrust', None),
                'cruise_sfc': engine.get('cruise_sfc', None),
                'ei_nox_to': engine.get('ei_nox_to', None),
                'ei_nox_co': engine.get('ei_nox_co', None),
                'ei_nox_app': engine.get('ei_nox_app', None),
                'ei_nox_idl': engine.get('ei_nox_idl', None),
                'ei_co_to': engine.get('ei_co_to', None),
                'ei_co_co': engine.get('ei_co_co', None),
                'ei_co_app': engine.get('ei_co_app', None),
                'ei_co_idl': engine.get('ei_co_idl', None)
            }
        except KeyError:
            logger.error(f"Engine data not found for engine type: {engine_type}")
            return {}

    def calculate_fuel_and_emissions(
        self, 
        flight: dict
        ) -> FlightPerformance:
        """
        Calculate fuel flow and emissions for a given flight.
        
        Args:
            flight (Flight): A flight object with relevant aircraft data.
        
        Returns:
            FlightPerformance: A flight performance object with fuel and emission data.
        """
        logger.debug(f"Calculating fuel flow and emissions for flight {flight['flight_id']}...")

        mapped_ac_type = self.aircraft_type_mapping.get(flight['aircraft_icao_code'], flight['aircraft_icao_code'])

        try:
            fuelflow = FuelFlow(ac=mapped_ac_type)
            emission = Emission(ac=mapped_ac_type)

            # Calculate fuel flow and emissions
            fuel_flow = fuelflow.enroute(
                mass=flight['aircraft_mtow_kg'],
                tas=flight['true_airspeed_ms'] * 1.94384,  # Convert m/s to knots
                alt=flight['altitude'] * 3.28084,           # Convert meters to feet
                vs=flight['vertical_speed'] * 196.85        # Convert m/s to feet/min
            )
            co2_flow = emission.co2(fuel_flow)
            h2o_flow = emission.h2o(fuel_flow)
            nox_flow = emission.nox(fuel_flow, flight['true_airspeed_ms'] * 1.94384, flight['altitude'] * 3.28084)
            co_flow = emission.co(fuel_flow, flight['true_airspeed_ms'] * 1.94384, flight['altitude'] * 3.28084)
            hc_flow = emission.hc(fuel_flow, flight['true_airspeed_ms'] * 1.94384, flight['altitude'] * 3.28084)
            soot_flow = emission.soot(fuel_flow)
            
            #cast the latitude and longitude to string (for hopsworks primary key requirements)
            flight['latitude'] = str(flight['latitude'])
            flight['longitude'] = str(flight['longitude'])
            
            # Create and return a FlightPerformance object
            return FlightPerformance(
                **flight,
                co2_flow=co2_flow,
                h2o_flow=h2o_flow,
                nox_flow=nox_flow,
                co_flow=co_flow,
                hc_flow=hc_flow,
                soot_flow=soot_flow,
                fuel_flow=fuel_flow,
            )
            
        except Exception as e:
            logger.error(f"Error calculating fuel flow and emissions for flight {flight['flight_id']}: {e}")
            
            #cast the latitude and longitude to string (for hopsworks primary key requirements)
            flight['latitude'] = str(flight['latitude'])
            flight['longitude'] = str(flight['longitude'])
            
            return FlightPerformance(**flight, fuel_flow=0, co2_flow=0, h2o_flow=0, nox_flow=0, co_flow=0, hc_flow=0, soot_flow=0)

    
if __name__ == '__main__':
    # Create an instance of the AddAircraftPerformances class
    openap = AddAircraftPerformances()

    # Load a sample flight data
    flight = Flight(
        aircraft_icao_code='a320',
        aircraft_iata_code='320',
        aircraft_mtow_kg=78000,
        aircraft_malw_kg=70000,
        aircraft_engine_class='turbofan',
        aircraft_num_engines=2,
        airline_iata_code='LH',
        airline_icao_code='DLH',
        airline_name='Lufthansa',
        altitude=35000,
        flight_level=350,
        arrival_airport_iata='JFK',
        arrival_airport_icao='KJFK',
        arrival_city='New York',
        current_flight_time=3600,
        departure_airport_iata='FRA',
        departure_airport_icao='EDDF',
        departure_city='Frankfurt',
        direction=90,
        flight_icao_number='DLH123',
        flight_number='LH123',
        flight_id='DLH123-2021-01-01',
        flight_status='enroute',
        horizontal_speed=450,
        isGround=False,
        latitude=40.7128,
        longitude=-74.0060,
        vertical_speed=0,
        true_airspeed_ms=220,
        departure_country='Germany',
        arrival_country='USA',
        route='EDDF..JFK',
        departure_airport_lat=50.0333,
        arrival_airport_lat=40.6413,
        departure_airport_long=8.5706,
        arrival_airport_long=73.7781
    )

    # Add aircraft performances to the flight data
    flight_performance = openap.add_aircraft_performances(flight)

    # Print the flight performance data
    logger.info(flight_performance)
    