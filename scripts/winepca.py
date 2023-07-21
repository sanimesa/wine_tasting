import openai
import os
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

def get_label1(wine_name):
    if wine_name.startswith('Cabernet'):
        return 'Cabernet'
    elif  wine_name.startswith('Merlot'):
        return 'Merlot'
    else:
        return 'Chardonnay'

def get_label(wine_name):
    if wine_name.startswith('Cabernet') or wine_name.startswith('Merlot'):
        return 'Red'
    else:
        return 'White'
    
def run_pca(df):
    column_names = ['feature_{}'.format(i) for i in range(1,1537)]
    print(len(column_names))
    print(df.shape)
    print(df.info())

    df['label'] = df['wine_name'].apply(get_label1)

    # print(df.head())

    # for index, row in df.iterrows():
    #     embedding = row["embedding"]
    #     print(type(embedding))
    #     arr = np.array(embedding)
    #     print(arr)
    #     break

    df2 = pd.DataFrame(np.array(df['embedding']).tolist(), columns=column_names, index= df.index)
    # df2 = pd.DataFrame(df['embedding'].str.split(','), columns=column_names, index= df.index)
    print(df2.head())

    x = df2.values
    x = StandardScaler().fit_transform(x) # normalizing the features
    print(x)
    print(x.shape)
    print(np.mean(x),np.std(x))

    normalised_df = pd.DataFrame(x,columns=column_names)
    print(normalised_df.tail())
    
    pca_df = PCA(n_components=2)
    principalComponents_df = pca_df.fit_transform(x)
    principal_Df = pd.DataFrame(data = principalComponents_df, columns = ['principal component 1', 'principal component 2'])
    print(principal_Df.tail())

    print('Explained variation per principal component: {}'.format(pca_df.explained_variance_ratio_))

    # plt.figure()
    plt.figure(figsize=(7,7))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Principal Component - 1',fontsize=10)
    plt.ylabel('Principal Component - 2',fontsize=10)
    plt.title("Principal Component Analysis of Wines",fontsize=12)
    targets = ['Cabernet', 'Merlot', 'Chardonnay']
    # targets = ['Red', 'White']
    colors = ['r', 'm', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = df['label'] == target
        plt.scatter(principal_Df.loc[indicesToKeep, 'principal component 1']
                , principal_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

    plt.legend(targets,prop={'size': 10})
    plt.axvline(x = 0.0, color = 'b', linestyle = '--')
    plt.show()

    # return principal_Df

    df_new = df[['wine_name', 'tasting_note', 'label']]
    df_merged = pd.concat([df_new, principal_Df], axis=1) #df_new.merge(principal_Df)
    
    print(df_merged.head())
    print(df_merged[(df_merged['principal component 1'] < 0) & (df_merged['label'] == 'Chardonnay')])

def plot_wine_clusters(principal_Df):
    # plt.figure()
    plt.figure(figsize=(7,7))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Principal Component - 1',fontsize=10)
    plt.ylabel('Principal Component - 2',fontsize=10)
    plt.title("Principal Component Analysis of Wines",fontsize=12)
    targets = ['Cabernet', 'Merlot', 'Chardonnay']
    # targets = ['Red', 'White']
    colors = ['r', 'm', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = df['label'] == target
        plt.scatter(principal_Df.loc[indicesToKeep, 'principal component 1']
                , principal_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

    plt.legend(targets,prop={'size': 10})
    plt.axvline(x = 0.0, color = 'b', linestyle = '--')
    plt.show()

async def get_wines():
    conn = await asyncpg.connect(os.getenv("NEONDB_CONNSTR"))
    await register_vector(conn)

    results = await conn.fetch(
        """
            SELECT wine_name, tasting_note, embedding
            FROM wines
        """
    )

    await conn.close()

    df = pd.DataFrame(results, columns=['wine_name', 'tasting_note', 'embedding'])
    print(df.head())

    return df


def main():
    df = asyncio.get_event_loop().run_until_complete(get_wines())
    run_pca(df)
    # plot_wine_clusters(principal_df)

if __name__ == '__main__':
    main()