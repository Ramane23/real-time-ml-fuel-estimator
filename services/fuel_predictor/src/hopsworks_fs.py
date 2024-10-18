from typing import List, Optional, Dict
import time
import uuid
from loguru import logger
import pandas as pd
import hopsworks
from quixstreams import Application
import json
from hsfs.feature_store import FeatureStore
from hsfs.client.exceptions import FeatureStoreException
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

from src.config import config

# This class is responsible for writing the flights data to a feature group in Hopsworks
class HopsworksFlightsWriter:
    """_This class is responsible for writing the flights data to a feature group in Hopsworks_"""
    
    def __init__(
        self,
        feature_group_name : str, 
        feature_group_version : int, 
        hopsworks_project_name : str, 
        hopsworks_api_key : str         
        ):
        
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version
        self.hopsworks_project_name = hopsworks_project_name
        self.hopsworks_api_key = hopsworks_api_key
    
    #Function that gets a pointer to the feature store
    def get_feature_store(self) -> FeatureStore:
        """Connects to Hopsworks and returns a pointer to the feature store

        Returns:
            hsfs.feature_store.FeatureStore: pointer to the feature store
        """
        # Log in to Hopsworks and get the project
        project = hopsworks.login(
            project= self.hopsworks_project_name,
            api_key_value= self.hopsworks_api_key
        )
        # Return the feature store for the project
        return project.get_feature_store()
    
    #Function that gets a pointer to the feature group
    def get_feature_group(
        self,
        feature_store: FeatureStore,
        live_or_historical: str
        ) -> FeatureGroup:
        """
        Returns (and possibly creates) the feature group we will be writing to.
        """
        
        if live_or_historical == 'historical':
            feature_group = feature_store.get_or_create_feature_group(
                name=self.feature_group_name,
                version=self.feature_group_version,
                description='historical Flights tracking data enriched with weather data',
                primary_key=[
                            'flight_id', 
                            'latitude',
                            'longitude',
                            'current_flight_time', 
                            'flight_level'
                        ],
                event_time='current_flight_time',
                #online_enabled=True, # Enable online feature serving
            )
        elif live_or_historical == 'live':
            feature_group = feature_store.get_or_create_feature_group(
                name=self.feature_group_name,
                version=self.feature_group_version,
                description='live Flights tracking data enriched with weather data',
                primary_key=[
                            'flight_id', 
                            'latitude',
                            'longitude',
                            'current_flight_time', 
                            'flight_level'
                        ],
                event_time='current_flight_time',
                online_enabled=True, # Enable online feature serving
            )

        return feature_group
    
    #Function that writes the data to the feature group
    def push_flight_data_to_feature_store(
        self,
        flight_data: List[dict],
        #online_or_offline: str,
    ) -> None:
        """
        Pushes the given `flight_data` to the feature store, writing it to the feature group
        with name `feature_group_name` and version `feature_group_version`.

        Args:
            feature_group_name (str): The name of the feature group to write to.
            feature_group_version (int): The version of the feature group to write to.
            flight_data (List[dict]): The flight data to write to the feature store.
            online_or_offline (str): Whether we are saving the `flight_data` to the online or offline
            feature group.

        Returns:
            None
        """
        
        # Get the feature store
        #feature_store = self.get_feature_store()

        # Get or create the feature group we will be saving flight data to
        flight_feature_group = self.get_feature_group(
            feature_store=self.get_feature_store(),
            live_or_historical = config.live_or_historical
        )

        # Transform the flight data (dict) into a pandas dataframe
        flight_data_df = pd.DataFrame(flight_data)
        
        # Write the flight data to the feature group
        flight_feature_group.insert(
            flight_data_df,
            #write_options={
                #'start_offline_materialization': True # we are telling the feature store to start copying the data to the offline database 
                #if online_or_offline == 'offline' #if the online_offline argument is set to offline
                #else False
            #},
        ) 
        
        return None
    

class FlightPrimaryKeyFetcher:
    def __init__(
        self, 
        kafka_topic_name: str, 
        kafka_broker_address: str, 
        kafka_consumer_group: str, 
        #live_or_batch: str = "live",
        create_new_consumer_group: Optional[bool] = True,
    ):
        self.kafka_topic_name = kafka_topic_name
        self.kafka_broker_address = kafka_broker_address
        self.kafka_consumer_group = kafka_consumer_group
        self.create_new_consumer_group = create_new_consumer_group
        self.flight_data_buffer = []  # Buffer to store flight data within the specified time window
        
    
    def _fetch_flight_data_from_kafka(self, last_n_minutes: int):
        """
        Polls the Kafka topic for flight data and stores it in a rolling buffer.
        Only data from the last `last_n_minutes` minutes is kept.
        """
        current_time = int(time.time())  # Current time in seconds
        end_time = current_time - 1800 # Start time is one quarter of an hour before the current time (to accommodate with the aviationedge API)
        start_time = end_time - last_n_minutes * 60  # Start time is `last_n_minutes` minutes before the end time
        #breakpoint()
        
        if self.create_new_consumer_group:
            # generate a unique consumer group name using uuid
            kafka_consumer_group = 'flights_with_apm_consumer_' + str(uuid.uuid4())
            logger.debug(f'New consumer group name: {kafka_consumer_group}')
            #breakpoint()
            
        offset_reset = "earliest"
        app = Application(
            broker_address=self.kafka_broker_address,
            consumer_group=self.kafka_consumer_group,
            auto_offset_reset=offset_reset,
            commit_interval=1.0  # Commit offsets every 1 second
        )
        logger.info("Application created")
        topic = app.topic(name=self.kafka_topic_name, value_serializer='json')
        
        #Ensure the flight data buffer is empty each time the function is called
        self.flight_data_buffer = []
        
        with app.get_consumer() as consumer:
            consumer.subscribe(topics=[topic.name])
            logger.info(f"Subscribed to topic {self.kafka_topic_name}")
            
            logger.debug(f"gathering fight data from the last {last_n_minutes} minutes")
            while True:
                msg = consumer.poll(100)  # Poll with a timeout of 100ms
                if msg is None:
                    logger.debug(f"No new messages available in the input topic {self.kafka_topic_name}")
                    break
                elif msg.error():
                    logger.error(f"Kafka error: {msg.error()}")
                    continue
                
                msg_value = json.loads(msg.value().decode('utf-8'))
                if all(k in msg_value for k in ['flight_id', 'latitude', 'longitude', 'current_flight_time', 'flight_level']):
                    flight_time = int(msg_value['current_flight_time'])
                    #breakpoint()
                    if flight_time < start_time:
                        logger.warning(f"The pooled flight point is before {start_time}, continuing the polling...")
                        
                    # If the data is within the one-hour window
                    elif flight_time in range(start_time, end_time):
                        self.flight_data_buffer.append(msg_value)  # Append new data
                        logger.info(f"Buffer size: {len(self.flight_data_buffer)}")
                        #breakpoint()
                    elif flight_time > end_time:
                        # If we hit data older than one hour before the start time, stop polling
                        logger.warning(f"The pooled flight point is beyond the {end_time}, stopping the polling...")
                        # I don't want the breaking to lead to an error in case flight_data_buffer is empty
                        # if the buffer is empty, we will just return an empty list of primary keys
                        if len(self.flight_data_buffer) == 0:
                            logger.warning("The flight data buffer is empty.")
                        else:
                            logger.info(f"All the flight data within the range {start_time} and {end_time} has been fetched")
                        break
                            
                            
    def get_primary_keys(self, last_n_minutes: int) -> List[Dict]:
        """
        Returns a list of primary keys of the flights produced in the last `last_n_minutes` minutes.
        
        Args:
            last_n_minutes (int): The number of minutes to go back in time.
        
        Returns:
            List[Dict]: A list of dictionaries with the primary keys.
        """
        # Ensure buffer is filled with the latest flight data from Kafka
        if not self.flight_data_buffer:
            logger.debug(f"Flight buffer is empty. Fetching data from Kafka for the last {last_n_minutes} minutes.")
            self._fetch_flight_data_from_kafka(last_n_minutes)  # Fetch data for the last `last_n_minutes` minutes
            #breakpoint()
        # If buffer is still empty, return an empty list
        if not self.flight_data_buffer:
            logger.error("No flight data available from Kafka.")
            return []
        
        # Generate the primary keys from the buffered flight data
        primary_keys = [
            {
                'flight_id': flight['flight_id'],
                'latitude': flight['latitude'],
                'longitude': flight['longitude'],
                'current_flight_time': flight['current_flight_time'],
                'flight_level': flight['flight_level']
            }
            for flight in self.flight_data_buffer
        ]
        #breakpoint()
        logger.info(f"Retrieved {len(primary_keys)} primary keys from the flight data buffer.")
        logger.debug(f"Primary keys: {primary_keys}")
        
        return primary_keys


#A class that reads the flights data from a feature group in Hopsworks
class HopsworksFlightsReader:
    """This class reads the flights data from a feature group in Hopsworks"""
    
    #instantiating the FlightPrimaryKeyFetcher class
    flight_primary_key_fetcher = FlightPrimaryKeyFetcher(
        config.kafka_topic_name,
        config.kafka_broker_address,
        config.kafka_consumer_group,
    )
    
    #Instantiating the class
    def __init__(
        self,
        feature_store : FeatureStore,
        feature_group_name : str, 
        feature_group_version : int ,
        feature_view_name : Optional[str], 
        feature_view_version : Optional[int] 
        ):
        
        self.feature_store = feature_store
        self.feature_group_name  = feature_group_name
        self.feature_group_version  = feature_group_version
        self.feature_view_name  = feature_view_name
        self.feature_view_version  = feature_view_version
    
    
    #Function to get the feature view
    def get_feature_view(self) -> FeatureView:
        """
        Returns the feature view object that reads data from the feature store
        """
        if self.feature_group_name is None:
            # We try to get the feature view without creating it.
            # If it does not exist, we will raise an error because we would
            # need the feature group info to create it.
            try:
                return self.feature_store.get_feature_view(
                    name=self.feature_view_name,
                    version=self.feature_view_version,
                )
            except Exception:
                raise ValueError(
                    'The feature group name and version must be provided if the feature view does not exist.'
                )
        
        # We have the feature group info, so we first get it
        feature_group = self.feature_store.get_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
        )

        # and we now create it if it does not exist
        feature_view = self.feature_store.get_or_create_feature_view(
            name=self.feature_view_name,
            version=self.feature_view_version,
            query=feature_group.select_all(),
        )
        # and if it already existed, we check that its feature group name and version match
        # the ones we have in `self.feature_group_name` and `self.feature_group_version`
        # otherwise we raise an error
        possibly_different_feature_group = \
            feature_view.get_parent_feature_groups().accessible[0]
        
        if possibly_different_feature_group.name != feature_group.name or \
            possibly_different_feature_group.version != feature_group.version:
            raise ValueError(
                'The feature view and feature group names and versions do not match.'
            )
        
        return feature_view
    
class GetFeaturesFromTheStore:
    
    def __init__(self):
        pass
    
    #Instantiate the HopsworksFlightsWriter class
    hopsworks_flights_writer = HopsworksFlightsWriter(
        config.feature_group_name, 
        config.feature_group_version, 
        config.hopsworks_project_name, 
        config.hopsworks_api_key
    )

    #Instantiate the HopsworksFlightsReader class
    hopsworks_flights_reader = HopsworksFlightsReader(
        hopsworks_flights_writer.get_feature_store(),
        hopsworks_flights_writer.feature_group_name,
        hopsworks_flights_writer.feature_group_version,
        config.feature_view_name,
        config.feature_view_version
    )

    #instantiating the FlightPrimaryKeyFetcher class
    flight_primary_key_fetcher = FlightPrimaryKeyFetcher(
        config.kafka_topic_name,
        config.kafka_broker_address,
        config.kafka_consumer_group,
    )
    
    #function to get the flights data from the feature store
    def get_features(
        self,
        live_or_historical: str,
        ) -> pd.DataFrame:
        """
        Fetches the features from the store and returns them as a pandas DataFrame.
        All the config parameters are read from the src.config module

        Args:
            None

        Returns:
            pd.DataFrame: The features as a pandas DataFrame sorted by timestamp (ascending)
        """
        logger.debug('Getting the feature view')
        #Get the feature view 
        feature_view = self.hopsworks_flights_reader.get_feature_view()   

        # For the moment, let's get all rows from this feature group
        if live_or_historical == 'historical':
            try:
                #retrieving the data from the offline feature store which is a structured store (like SQLite)
                features: pd.DataFrame = feature_view.get_batch_data()
                #breakpoint()
            except FeatureStoreException:
                # breakpoint()
                # retry the call with the use_hive option. This is what Hopsworks recommends
                features: pd.DataFrame = feature_view.get_batch_data(read_options={"use_hive": True})
                #breakpoint()
        elif live_or_historical == 'live':
            # we fetch from the online feature store.
            # we need to build this list of dictionaries with the primary keys because the online feature store does not support batch reads
            #it is designed for fast online reads of individual rows and require the primary keys to be passed
            features : pd.DataFrame = feature_view.get_feature_vectors(
                #entry = hopsworks_flights_reader.get_primary_keys(
                    #kafka_topic_name=config.kafka_topic_name,
                    #kafka_broker_address=config.kafka_broker_address,
                    #kafka_consumer_group=config.kafka_consumer_group,
                    #live_or_batch=config.live_or_batch
                    #),
                entry = self.flight_primary_key_fetcher.get_primary_keys(last_n_minutes=config.last_n_minutes),
                return_type="pandas"
                )
            #breakpoint()    
        # sort the features by timestamp (ascending)
        #features = features.sort_values(by='current_flight_time')
        features.sort_values(by='current_flight_time', inplace=True)
        #breakpoint()
        return features
    
if __name__ == '__main__':
    #Instantiate the GetFeaturesFromTheStore class
    get_features_from_the_store = GetFeaturesFromTheStore()
    #Get the features from the feature store
    features = get_features_from_the_store.get_features(live_or_historical=config.live_or_historical)
    #print the features
    features.head()