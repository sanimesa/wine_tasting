# Wine Tasting

Use LLMs to analyze wine reviews and test vector databases. 

Example code used in the article: [Raise a roast to the Vector Databases](https://medium.com/@shuvro_25220/raise-a-toast-to-the-vector-databases-fd2cfae60549)

This repo contains code that scrapes wine reviews from Wine Spectator magazine and then uses a sentence embadding model to embed the tasting notes.
The resulting dataset of wines with their embeddings allow us to perform semantic (Vector) search and find wines with characteristics similar to a preferred wine.

A sample PCA analysis on the vectors shows how Wines from different varieties of grapes are closely clustered together:


![image](https://github.com/user-attachments/assets/e893ce30-bb45-4e0e-847d-c64d2d794c5f)



