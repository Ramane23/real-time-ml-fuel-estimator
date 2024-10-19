#!/bin/bash
export KAFKA_BROKER_ADDRESS="localhost:19092"
export KAFKA_TOPIC_NAME="live_flights_with_apm"
export KAFKA_CONSUMER_GROUP="live_flights_with_apm_consumer_primary_keys"
export LIVE_OR_HISTORICAL="live"
export HOPSWORKS_API_KEY="MxkTFcD7JWLb15l0.T6u5GKvw6thyaxkk6qLSYXcTCZr8efDc6yfuJehsaDxeUHBxebGbuKyfkJTib8b6"
export HOPSWORKS_PROJECT_NAME="Ramane"
export FEATURE_GROUP_NAME="live_flights_tracking_with_apm_data"
export FEATURE_GROUP_VERSION=1
export FEATURE_VIEW_NAME="live_flights_tracking_with_apm_view"
export FEATURE_VIEW_VERSION=1
export LAST_N_MINUTES=10
export COMET_ML_API_KEY="dm9Y7P3rP2HsnOJ4RIvrW6ULw"
export COMET_ML_PROJECT_NAME="real-time-ml-fuel-estimator"
export COMET_ML_WORKSPACE="ramane23"