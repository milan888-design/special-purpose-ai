CREATE EXTENSION vector;
worked

-- Create a table with a vector column
CREATE TABLE articles (
    id serial PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding VECTOR(1536) -- Adjust dimensions to match your embedding model (e.g., Google's models)
);

-- Insert data with embeddings (you'll generate these with an embedding model)
INSERT INTO articles (title, content, embedding) VALUES
('AI Revolution', 'Artificial intelligence is transforming industries', '[0.1, 0.2, 0.3]');

-- Perform a similarity search (using cosine distance as an example)
-- Replace [0.4, 0.5, 0.6, ...] with the embedding of your query
SELECT title, content
FROM articles
ORDER BY embedding <=> '[0.4, 0.5, 0.6, ...]'
LIMIT 5;

CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));

INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');

SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;

--28may25
CREATE TABLE items2 (id bigserial PRIMARY KEY, embedding vector(3));

INSERT INTO items2 (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');

SELECT * FROM items2 ORDER BY embedding <-> '[3,1,2]' LIMIT 5;

CREATE TABLE documents (
    id int PRIMARY KEY,
    title text NOT NULL,
    content TEXT NOT NULL
);
--worked

-- Create document_embeddings table
CREATE TABLE document_embeddings (
    id int PRIMARY KEY,
    embedding vector(1536) NOT NULL
);
--worked

CREATE INDEX document_embeddings_embedding_idx ON document_embeddings USING hnsw (embedding vector_l2_ops);
--worked

-- Insert documents into documents table
INSERT INTO documents VALUES ('1', 'pgvector', 'pgvector is a PostgreSQL extension that provides support for vector similarity search and nearest neighbor search in SQL.');
INSERT INTO documents VALUES ('2', 'pg_similarity', 'pg_similarity is a PostgreSQL extension that provides similarity and distance operators for vector columns.');
INSERT INTO documents VALUES ('3', 'pg_trgm', 'pg_trgm is a PostgreSQL extension that provides functions and operators for determining the similarity of alphanumeric text based on trigram matching.');
INSERT INTO documents VALUES ('4', 'pg_prewarm', 'pg_prewarm is a PostgreSQL extension that provides functions for prewarming relation data into the PostgreSQL buffer cache.');
--worked


INSERT INTO documents VALUES ('5', 'question1', 'how do you become expert in Epic game');
INSERT INTO documents VALUES ('6', 'question2', 'give me steps to learn Epic game');
INSERT INTO documents VALUES ('7', 'prompt1', 'Install Epic game, watch relevant youtube gamers, practice');
INSERT INTO documents VALUES ('8', 'prompt2', 'hire Epic game expert for private tutoring');
--worked

delete from document_embeddings

SELECT * FROM public.document_embeddings ORDER BY id ASC 

WITH pgv AS (
    SELECT embedding
      FROM document_embeddings JOIN documents USING (id)
     WHERE title = 'question1'
)
SELECT title, content
  FROM document_embeddings
  JOIN documents USING (id)
 WHERE embedding <-> (SELECT embedding FROM pgv) < 0.6;
--.6 results in prompt1 and prompt2 both. while .5 gives only since it does not have word expert in it.
--higher the number from .5 to .6 to .7. less accurate it becomes
--.4 returns the exact as the question and not other. that is almost equal to 
--.6 is better. This ia about multiple word match compare to string match 
-- where prompt like '%epic%' would fine anythin with that. 
--one can 