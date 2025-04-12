import os
import pandas as pd
import asyncio
import cybotrade_datasource
from datetime import datetime, timezone


# API_KEY = os.environ["API_KEY"]


async def main():
    data = await cybotrade_datasource.query_paginated(
        api_key="CFyapwMlPYqScPa2s4LuDGvAKKWhrDWnj7EhNj4BvtRxmERA", 
        topic='cryptoquant|btc/inter-entity-flows/miner-to-miner?from_miner=f2pool&to_miner=all_miner&window=hour', 
        start_time=datetime(year=2024, month=1, day=1, tzinfo=timezone.utc),
        end_time=datetime(year=2025, month=1, day=1, tzinfo=timezone.utc)
    )
    df = pd.DataFrame(data)
    print(df)
    

asyncio.run(main())