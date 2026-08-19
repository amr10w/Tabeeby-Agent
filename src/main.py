from stores.llm.LLMProviderFactory import LLMProviderFactory
from helpers.config import get_settings
from stores.llm.LLMEnums import LLMEnums
from dotenv import load_dotenv
from ollama import ChatResponse

load_dotenv()
config=get_settings()
factory=LLMProviderFactory(config=config)
client = factory.create(LLMEnums.OLLAMA.value)

client.set_generation_model(config.GENERATION_MODEL_ID)
response:ChatResponse=client.generate_response("Hi",None)

print(response.message.content)
