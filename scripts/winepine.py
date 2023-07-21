import openai
import os
import sys 
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import tiktoken
import asyncio
import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from openai.embeddings_utils import get_embedding
import pinecone 

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def get_label(wine_name):
    if wine_name.startswith('Cabernet'):
        return 'Cabernet'
    elif  wine_name.startswith('Merlot'):
        return 'Merlot'
    elif  wine_name.startswith('Chardonnay'):
        return 'Chardonnay'
    else:
        return 'Other'
        
  
async def get_wines():
    conn = await asyncpg.connect(os.getenv("NEONDB_CONNSTR"))
    await register_vector(conn)

    results = await conn.fetch(
        """
            SELECT wine_name, vineyard, 
            cast(split_part(replace(price, '$', ''), ' ', 1) as int) as price, 
            tasting_note, embedding
            FROM wines
        """
    )

    await conn.close()

    df = pd.DataFrame(results, columns=['wine_name', 'vineyard', 'price', 'tasting_note', 'embedding'])
    print(df.head())

    payload = [] 

    for idx, row in df.iterrows():
        payload.append(tuple(['wine_'+str(idx), np.array(row['embedding']).tolist(), 
                              {"wine_name": row["wine_name"], "varietal": get_label(row["wine_name"]), "vineyard": row["vineyard"], "price": row["price"], "tasting_note": row["tasting_note"]}]))

    print(len(payload), sys. getsizeof(payload) )

    return payload

async def getTestWine():
    conn = await asyncpg.connect(os.getenv("NEONDB_CONNSTR"))
    await register_vector(conn)

    results = await conn.fetch("SELECT * from test_wine")

    await conn.close()

    print(results[0]['embedding'])

    return np.array(results[0]['embedding']).tolist()

def updatePineconeIndex(payload):
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp-free")

    # index.upsert(payload) #produces an API limit exception as the request size exceeeds 2MB

    with pinecone.Index('wine-tasting', pool_threads=3) as index:
        # Send requests in parallel
        async_results = [
            index.upsert(vectors=ids_vectors_chunk, async_req=True)
            for ids_vectors_chunk in chunks(payload, batch_size=100)
        ]
        # Wait for and retrieve responses (this raises in case of error)
        results = [async_result.get() for async_result in async_results]    

    print(results)

def queryPineconeIndex(payload):
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp-free")

    index = pinecone.Index("wine-tasting")

    results = index.query(
            vector=payload,
            top_k=3,
            filter={
                "price": {"$lt": 100},
                "varietal": "Cabernet"
            },
            include_values=False,
            include_metadata=True
        )

    print(results)

    # for match in results['matches']:
    #     print(match['id'], match['namespace'], match['score'])

def main():
    #build the pinecone index from the set of wine reviews
    # payload = asyncio.get_event_loop().run_until_complete(get_wines())
    # updatePineconeIndex(payload)

    query_payload = asyncio.get_event_loop().run_until_complete(getTestWine())
    queryPineconeIndex(query_payload)

if __name__ == '__main__':
    main()