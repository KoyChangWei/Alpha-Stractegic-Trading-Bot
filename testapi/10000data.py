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
        limit=10000
    )
    df = pd.DataFrame(data)
    print(df)
    

asyncio.run(main())