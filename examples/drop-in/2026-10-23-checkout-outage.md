# Incident: Checkout errors (2026-10-23) - ONGOING

Severity: SEV2. Started 14:05 UTC.

About 12% of checkout attempts fail with a timeout from the payments service. Cause under investigation: a connection pool limit reached after this morning's deploy of payments v4.18.

Incident commander: Avi Mizrahi. Mitigation in progress: rolling back payments to v4.17. Next update at 15:30 UTC.
