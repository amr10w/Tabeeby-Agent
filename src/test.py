from stores.llm.LLMProviderFactory import LLMProviderFactory,LLMEnums
from stores.llm.LLMEnums import LLMEnums

from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.vectordb.VectorDBEnums import VectorDBEnums

from agent import Agent

from helpers import get_settings

from tools import DoctorTools,read_file,write_file,list_files,create_directory,web_search

from dotenv import load_dotenv

load_dotenv()
config = get_settings()

factory = LLMProviderFactory(config=config)
factory_db = VectorDBProviderFactory(config=config)

client = factory.create(LLMEnums.OLLAMA.value)
client_embed = factory.create(LLMEnums.OLLAMAE.value)

client.set_generation_model(config.GENERATION_MODEL_ID)
client_embed.set_embedding_model(
    config.EMBEDDING_MODEL_ID,
    config.EMBEDDING_MODEL_SIZE)

client_db = factory_db.create(VectorDBEnums.QDRANT.value)

client_db.connect()
doctor_tool=DoctorTools(embedding_client=client_embed,
            vectordb_client=client_db)

tools = [doctor_tool.search_doctors,read_file,write_file,list_files,create_directory,web_search]
agent = Agent(client=client,
              tools=tools)


query = "Hi"
answer,meesages,cost=agent.run(query)
print(answer)

while True:

    query = input("Enter a query or exit\n")
    if query.strip().lower() == "exit":
        break

    answer,meesages,cost=agent.run(query,chat_history=meesages,total_cost=cost)
    print(answer)


print(meesages)
print(f"\n\ncost: {cost}")


