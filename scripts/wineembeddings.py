import openai
import pandas as pd
import json
import os
import tiktoken
import asyncio
import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from openai.embeddings_utils import get_embedding

openai.api_key = os.getenv("OPENAI_API_KEY") 

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

test_text = 'Dark, firm and focused, with fine, tightly wound tannins surrounding a core of black cherry, bay leaf, licorice and black pepper flavors, lingering intently on the expressive, graceful finish. '
test_text1 = 'A well-structured red, this features lightly chewy tannins that frame flavors of damson plum preserves, ground coffee, fig paste and bittersweet cocoa. This is dark and brooding, with a smoky mineral sublayer as well as hints of dried herb and kirsch on the finish.'

# v = get_embedding(test_text1, engine=embedding_model)
# print(v)
# print(len(v))

def build_embeddings(file_name):
    with open(f"c:/temp/{file_name}", "r") as file:
        content = file.read()
        wines  = json.loads(content)

        # wines = wines[1:5]
        print(file.name, 'contains ', len(wines), ' reviews')

        for wine in wines:
            print(wine)
            embedding = get_embedding(wine['tasting_note'], engine=embedding_model)
            wine['embedding'] = embedding

    return wines


async def update_db(wines):
    conn = await asyncpg.connect(os.getenv("NEONDB_CONNSTR"))
    await register_vector(conn)

    # Store all the generated embeddings back into the database.
    for wine in wines:
        await conn.execute(
            "INSERT INTO wines (vineyard, wine_name, price, tasting_note, embedding) VALUES ($1, $2, $3, $4, $5)",
            wine['vineyard'],
            wine['wine_name'],
            wine['price'],
            wine['tasting_note'],
            np.array(wine['embedding']),
        )

    await conn.close()


# # Run the SQL commands now.
# await main()  # type: ignore

async def find_similar_wines(update_test=False):
    #https://truemythwinery.com/wines/true-myth-cabernet-sauvignon/ 
    test_note = 'full of polished aromas of blueberry, cherry and vanilla, leading to flavors of dark red fruits, black currants and hints of pepper, mocha and caramelized oak. Rich yet smooth, '
    conn = await asyncpg.connect(os.getenv("NEONDB_CONNSTR"))
    await register_vector(conn)

    similarity_threshold = 0.1
    num_matches = 10
    qe = get_embedding(test_note, engine=embedding_model)
    print(qe);

    if update_test: 
        await conn.execute(
            "INSERT INTO test_wine (id, wine_name, tasting_note, embedding) VALUES (1, $1, $2, $3)",
            'Cabernet Sauvignon True Myth 2020',
            test_note,
            np.array(qe),
        )

    # Find similar products to the query using cosine similarity search 
    # over all vector embeddings. This new feature is provided by `pgvector`.
    results = await conn.fetch(
        """
            SELECT vineyard, wine_name, price, tasting_note, 1 - (embedding <=> $1) AS similarity
            FROM wines
            WHERE 1 - (embedding <=> $1) > $2
            ORDER BY similarity DESC
            LIMIT $3
        """,
        qe,
        similarity_threshold,
        num_matches
    )

    if len(results) == 0:
        raise Exception("Did not find any results. Adjust the query parameters.")

    for r in results:
        print(r['wine_name'], r['price'], r['vineyard'], r['similarity'])


    await conn.close()

    df = pd.DataFrame(results, columns=['wine_name', 'price', 'vineyard', 'tasting_note', 'similarity'])
    print(df.head())

    # html = df.to_html(index=False)
    # with open('c:/temp/most_dissimilar_wines.html', 'w') as fo:
    #     fo.write(html)


def main():
    # wines_w_embeddings = build_embeddings('merlot_output_1.json')
    # asyncio.get_event_loop().run_until_complete(update_db(wines_w_embeddings))

    asyncio.get_event_loop().run_until_complete(find_similar_wines(update_test=False))

if __name__ == '__main__':
    main()