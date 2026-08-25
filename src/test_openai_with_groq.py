"""Test OpenAIProvider with Groq's OpenAI-compatible API."""

import os
from dotenv import load_dotenv
load_dotenv()

from helpers import get_settings
from stores.llm.LLMEnums import LLMEnums, OpenAIEnums
from stores.llm.providers.OpenAIProvider import OpenAIProvider
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.vectordb.VectorDBEnums import VectorDBEnums
from tools import DoctorTools, web_search
from agent import Agent


def test_openai_provider_with_groq():
    settings = get_settings()

    groq_api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    groq_api_url = settings.GROQ_API_URL or os.getenv("GROQ_API_URL")

    print("=== 1. Initializing OpenAIProvider with Groq API endpoint ===")
    provider = OpenAIProvider(
        api_key=groq_api_key,
        api_url=groq_api_url,
    )
    provider.set_generation_model("openai/gpt-oss-20b")
    print(provider)
    print("✓ OpenAIProvider initialized successfully")

    print("\n=== 2. Testing Simple Generation ===")
    resp = provider.generate_response("Reply with one word: Success")
    assert resp is not None, "generate_response returned None"
    assert resp.message is not None, "resp.message is None"
    print(f"✓ Response Content: {resp.message.content.strip()}")
    print(f"✓ Token usage: prompt={resp.prompt_eval_count}, completion={resp.eval_count}")
    assert resp.prompt_eval_count > 0, "Expected prompt tokens > 0"

    print("\n=== 3. Testing Tool Calling ===")
    def get_clinic_info(city: str) -> str:
        """Get clinic info for a city.
        Args:
            city: City name.
        """
        return f"Clinics open in {city}: Central Health Clinic"

    resp_tool = provider.generate_response(
        prompt="What clinic is open in Cairo?",
        tools=[get_clinic_info],
    )
    assert resp_tool is not None
    assert resp_tool.message is not None
    assert resp_tool.message.tool_calls is not None, "Expected tool_calls"
    assert len(resp_tool.message.tool_calls) > 0, "Expected at least 1 tool call"
    tc = resp_tool.message.tool_calls[0]
    print(f"✓ Tool call received: name={tc.function.name}, args={tc.function.arguments}")
    assert tc.function.name == "get_clinic_info"

    print("\n=== 4. Testing End-to-End Agent with OpenAIProvider ===")
    # Using Gemini embedding provider for vector search since Groq doesn't host embeddings
    from stores.llm.LLMProviderFactory import LLMProviderFactory
    factory = LLMProviderFactory(config=settings)
    embed_provider = factory.create(LLMEnums.GEMINIE.value)
    embed_provider.set_embedding_model("gemini-embedding-001", 1024)

    factory_db = VectorDBProviderFactory(config=settings)
    client_db = factory_db.create(VectorDBEnums.QDRANT.value)
    
    client_db.connect()
    print(client_db)

    # Check connection explicitly
    if not client_db.client:
        print("❌ Qdrant is locked by another process!")
    else:
        print(f"✅ Qdrant connected. Collections: {client_db.list_all_collections()}")


    doctor_tool = DoctorTools(
        embedding_client=embed_provider,
        vectordb_client=client_db
    )

    tools = [doctor_tool.search_doctors, web_search]
    agent = Agent(
        client=provider,
        tools=tools,
        max_iterations=5
    )

    query = "search me a doctor for leg pain use tool, any where , any price"
    answer, messages, cost = agent.run(query)
    print(f"✓ Agent Run finished. Cost: ${cost:.6f}")
    print(f"  Final Response (first 200 chars): {answer}...")
    assert len(messages) > 1, "Expected message history"
    assert answer and len(answer) > 0, "Expected non-empty answer"


if __name__ == "__main__":
    test_openai_provider_with_groq()
    print("\n==========================================")
    print("ALL OPENAI PROVIDER TESTS PASSED (100%)")
    print("==========================================")
