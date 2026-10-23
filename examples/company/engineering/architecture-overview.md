# Architecture Overview

## Order platform

The orders service is written in Go and stores data in PostgreSQL 16. Events are published to Kafka on the topic orders.v2. The checkout flow calls the payments service, which wraps the Stripe API.

## Fleet platform

Robots report telemetry every 5 seconds over MQTT to the fleet gateway. Telemetry is stored in TimescaleDB with 90 days of retention.

## Deployments

All services deploy through the CI pipeline with a canary stage that receives 5% of traffic for 20 minutes before full rollout.
