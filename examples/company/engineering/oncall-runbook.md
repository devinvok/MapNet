# On-call Runbook

## Rotation

The on-call rotation changes every Monday at 10:00. Each engineer is primary for one week, then secondary the following week. The schedule lives in the pager tool.

## Severity levels

- SEV1: customers cannot place orders or robots are offline. Page primary and secondary immediately; open an incident channel within 5 minutes.
- SEV2: degraded performance for some customers. Primary responds within 30 minutes.
- SEV3: internal impact only. Handle during business hours.

## After an incident

A postmortem is required for every SEV1 and SEV2, published within 5 business days.
