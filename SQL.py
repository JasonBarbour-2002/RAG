import subprocess
import time
import psycopg
import numpy as np
from tqdm.notebook import tqdm

class SQL:
    def __init__(self):
        # start PostgreSQL service
        subprocess.run(["brew", "services", "start", "postgresql"], check=True)
        time.sleep(1) # Wait to ensure it's fully started
        

    def connect(self, db_name="RAG"):
        # Connect to PostgreSQL
        #Create DB if it doesn't exist
        self.db_name = db_name
        subprocess.run(["createdb", db_name], check=False)
        self.conn = psycopg.connect(
            dbname=db_name,
            user="jasonbarbour",
            host="localhost",
            port=5432,
        )
        self.conn.autocommit = True

            
    def add_vectors(self, embedding):
        # Add vectors to the database
        with self.conn.cursor() as cursor:
            self.dim = embedding.shape[0]
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # check if the table exists
            self.table_name = f"vecs{self.dim}"
            cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{self.table_name}');")
            exists = cursor.fetchone()[0]
            if not exists:
                cursor.execute(f"CREATE TABLE {self.table_name} (id SERIAL PRIMARY KEY, embedding VECTOR({self.dim}));")
                for i in tqdm(range(embedding.shape[1])):
                    cursor.execute(f"INSERT INTO {self.table_name} (embedding) VALUES (%s);", (embedding[:,i].tolist(),))
            self.conn.commit()
            
    def ivfflat(self, query, k=5):
        # Create an IVFFLAT index
        with self.conn.cursor() as cursor:
            # check if index exists
            cursor.execute(f"SELECT * FROM pg_indexes WHERE tablename = '{self.table_name}' AND indexname = '{self.table_name}_embedding_idx';")
            exists = cursor.fetchone()
            if not exists:
                cursor.execute(f"CREATE INDEX ON {self.table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            cursor.execute(f"""
                SELECT id, embedding <=> %s::vector AS distance
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;""",
                (query.tolist(), query.tolist(), k)
            )
            results = cursor.fetchall()
            # convert back to numpy array
            ids = np.zeros(len(results), dtype=int)
            distances = np.zeros(len(results))
            for i, (id_num, dist) in enumerate(results):
                ids[i] = id_num - 1
                distances[i] = dist
            return ids, distances
        
    def hnsw(self, query, k=5):
        # Create an HNSW index
        with self.conn.cursor() as cursor:
            # check if index exists
            cursor.execute(f"SELECT * FROM pg_indexes WHERE tablename = '{self.table_name}' AND indexname = '{self.table_name}_embedding_idx';")
            exists = cursor.fetchone()
            if not exists:
                cursor.execute(f"CREATE INDEX ON {self.table_name} USING hnsw (embedding vector_cosine_ops);")
            cursor.execute(f"""
                SELECT id, embedding <=> %s::vector AS distance
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;""",
                (query.tolist(), query.tolist(), k)
            )
            results = cursor.fetchall()
            # convert back to numpy array
            ids = np.zeros(len(results), dtype=int) 
            distances = np.zeros(len(results))
            for i, (id_num, dist) in enumerate(results):
                ids[i] = id_num - 1
                distances[i] = dist                
            return ids, distances
    
    def close(self):
        # Close the connection
        self.conn.close()
        subprocess.run(["brew", "services", "stop", "postgresql"], check=True)
        time.sleep(1)  # Wait to ensure it's fully stopped
        print("PostgreSQL service stopped.")
        


if __name__ == '__main__':
    # with SQL() as sql:
    #     sql.test_db()
    
    # Test vector search 
    embedding = np.random.rand(4, 5).astype(np.float32)
    
    target = np.array([5, 7, 2, 6], dtype=np.float32)
    
    sql = SQL()
    sql.connect(db_name="test")
    sql.add_vectors(embedding)
    answers, distances = sql.ivfflat(target)
    print(f"answers: {answers}")
    print(f"distances: {distances}")
    
    
    answers, distances = sql.hnsw(target)
    print(f"answers: {answers}")
    print(f"distances: {distances}")
    sql.close()
    
    