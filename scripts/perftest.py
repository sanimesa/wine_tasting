import openai
import pandas as pd
import json
import os
import tiktoken
import asyncio
import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from sentence_transformers import SentenceTransformer
# >>> sentences = ["This is an example sentence", "Each sentence is converted"]
# >>> model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

async def read_data(filename):
    chunksize = 1000
    iter = 0

    for chunk in pd.read_csv(filename, chunksize=chunksize):
        iter +=1
        if iter < 464:
            print('skipping batch: ', iter)
            continue

        chunk_part = chunk[['fdc_id', 'brand_owner', 'branded_food_category', 'ingredients']]
        chunk_part.reset_index(drop=True, inplace=True)
        chunk_part.fillna('', inplace=True)
        # chunk_part.to_csv(f'c:/temp/datapart_chunk{iter}.csv')
        # print(chunk_part.shape)
        # lst = chunk_part['ingredients'].tolist()
        # lst1 = [type(ele) for ele in lst]
        # print(lst1)
        embeddings = build_embeddings(chunk_part['ingredients'].tolist())

        # print(len(embeddings))
        # print(embeddings.tolist())
        # print(type(embeddings))

        # em_series = pd.Series(embeddings.flatten())
        df_em = pd.DataFrame({'embedding': embeddings.tolist()})
        # print(df_em.shape)

        df = pd.concat([chunk_part, df_em], axis=1, ignore_index=True)
        df.to_csv(f'c:/temp/datapart{iter}.csv')
        df.fillna('', inplace=True)

        print(chunk_part.shape, df_em.shape, df.shape)
        # # df = chunk_part.append(np.array(em_series), ignore_index=True)

        data = [] 
        for index, row in df.iterrows():
            # print(tuple(row))
            data.append(tuple(row))

        # print(len(data))
        # print(df1.shape)

        print('updating batch: ', iter)
        await update_db(data)      
        # print(df)
        # break

def build_embeddings(sentences):
    return model.encode(sentences)

async def update_db(payload):
    conn = await asyncpg.connect(os.getenv("NEONDB_CONNSTR"))
    await register_vector(conn)

    await conn.copy_records_to_table('branded_food', records=payload)

# create table branded_food (
#   fdc_id INT,
#   brand_owner VARCHAR(1024),
#   branded_food_category VARCHAR(1024),
#   ingredients text,
#   embedding vector(384)
# );
    
    await conn.close()

# async def find_similar_wines(update_test=False):
#     #https://truemythwinery.com/wines/true-myth-cabernet-sauvignon/ 
#     test_note = 'full of polished aromas of blueberry, cherry and vanilla, leading to flavors of dark red fruits, black currants and hints of pepper, mocha and caramelized oak. Rich yet smooth, '
#     conn = await asyncpg.connect(os.getenv("NEONDB_CONNSTR"))
#     await register_vector(conn)

#     similarity_threshold = 0.1
#     num_matches = 10
#     qe = get_embedding(test_note, engine=embedding_model)
#     print(qe);

#     if update_test: 
#         await conn.execute(
#             "INSERT INTO test_wine (id, wine_name, tasting_note, embedding) VALUES (1, $1, $2, $3)",
#             'Cabernet Sauvignon True Myth 2020',
#             test_note,
#             np.array(qe),
#         )

#     # Find similar products to the query using cosine similarity search
#     # over all vector embeddings. This new feature is provided by `pgvector`.
#     results = await conn.fetch(
#         """
#             SELECT vineyard, wine_name, price, tasting_note, 1 - (embedding <=> $1) AS similarity
#             FROM wines
#             WHERE 1 - (embedding <=> $1) > $2
#             ORDER BY similarity ASC
#             LIMIT $3
#         """,
#         qe,
#         similarity_threshold,
#         num_matches
#     )

#     if len(results) == 0:
#         raise Exception("Did not find any results. Adjust the query parameters.")

#     for r in results:
#         print(r['wine_name'], r['price'], r['vineyard'], r['similarity'])


#     await conn.close()

#     df = pd.DataFrame(results, columns=['wine_name', 'price', 'vineyard', 'tasting_note', 'similarity'])
#     print(df.head())

#     # html = df.to_html(index=False)
#     # with open('c:/temp/most_dissimilar_wines.html', 'w') as fo:
#     #     fo.write(html)


def main():
    from timeit import default_timer as timer
    start = timer()
    # wines_w_embeddings = build_embeddings('merlot_output_1.json')
    # asyncio.get_event_loop().run_until_complete(update_db(wines_w_embeddings))

    #https://fdc.nal.usda.gov/download-datasets.html 
    filename = 'C:/Temp/FoodData_Central_branded_food_csv_2023-04-20/branded apr 2023/branded_food.csv'
    asyncio.get_event_loop().run_until_complete(read_data(filename))

    end = timer()
    print("total time taken: ", end - start)

if __name__ == '__main__':
    main()

