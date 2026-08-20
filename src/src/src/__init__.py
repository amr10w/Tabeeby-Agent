import ollama

single=ollama.embed(
    model="qwen3-embedding",
    input='The quick brown fox jumps over the lazy dog.',
    dimensions=1024
)
embeddings=single['embeddings']
print(len(single['embeddings'][0]))  
print(len(single['embeddings']))  

print(embeddings[0])