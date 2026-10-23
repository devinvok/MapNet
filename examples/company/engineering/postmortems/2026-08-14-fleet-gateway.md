# Postmortem: Fleet gateway outage (2026-08-14)

Severity: SEV1. Duration: 47 minutes. Robots affected: about 1,900.

## What happened

A certificate on the fleet gateway expired at 03:12 UTC. Robots could not reconnect over MQTT and went into safe mode.

## Root cause

Certificate renewal was a manual task and the reminder went to a former employee.

## Actions

Automate certificate renewal with a 30-day expiry alert (owner: Noa Levi, done). Add the gateway certificate to the weekly security scan.
