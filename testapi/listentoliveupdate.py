import os
import pandas as pd
import asyncio
import cybotrade_datasource
from datetime import datetime, timezone


# API_KEY = os.environ["API_KEY"]


async def main():
    stream = await cybotrade_datasource.stream(
        api_key="CFyapwMlPYqScPa2s4LuDGvAKKWhrDWnj7EhNj4BvtRxmERA",
        topics=[
            'cryptoquant|btc/inter-entity-flows/miner-to-miner?from_miner=f2pool&to_miner=all_miner&window=hour',
            'cryptoquant|btc/market-data/liquidations?exchange=deribit&window=min',
        ],
    )
    async for msg in stream:
        print(msg)
    

asyncio.run(main())