
## Dataset Source and Provenance

- Data provider: Dukascopy
- Download tool: dukascopy-node
- Instrument: xomususd
- Underlying company: Exxon Mobil Corporation
- Requested timeframe: 5 minutes
- Price type: Bid (default CLI setting)
- Requested period: 2017-01-26 to 2026-10-01
- File format: CSV
- Volume requested: No

## Original Download Command

npx dukascopy-node -i xomususd -from 2017-01-26 -to 2026-10-01 -t m5 -f csv

## Source Documentation

https://www.dukascopy-node.app/instrument/xomususd

https://www.dukascopy-node.app/config/cli

## Verification Status

Download method confirmed by the user and consistent
with the official instrument and CLI documentation.

Exact file contents have not been independently
re-downloaded and compared byte-for-byte.

The dataset contains Bid OHLC quotations, not
verified official NYSE transaction prices.

Known temporal gaps and coverage limitations remain.
