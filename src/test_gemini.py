"""Test script for GeminiProvider and Gemini integration."""

import os
from dotenv import load_dotenv

from helpers import get_settings
from stores.llm.LLMEnums import LLMEnums, GeminiEnums
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.vectordb.VectorDBEnums import VectorDBEnums
from tools import DoctorTools, web_search
from agent import Agent


def test_factory_and_enums(config):
    print("=== Testing Factory & Enums ===")
    assert LLMEnums.GEMINI.value == "GEMINI"
    assert LLMEnums.GEMINIE.value == "GEMINIE"
    assert GeminiEnums.SYSTEM.value == "system"
    assert GeminiEnums.USER.value == "user"
    assert GeminiEnums.ASSISTANT.value == "assistant"
    assert GeminiEnums.TOOL.value == "tool"

    factory = LLMProviderFactory(config=config)
    gen_provider = factory.create(LLMEnums.GEMINI.value)
    embed_provider = factory.create(LLMEnums.GEMINIE.value)

    assert gen_provider is not None, "Failed to create Gemini generation provider"
    assert embed_provider is not None, "Failed to create Gemini embedding provider"
    print("✓ Factory and Enums verified successfully")
    return gen_provider, embed_provider


def test_embedding(embed_provider):
    print("\n=== Testing Gemini Embedding ===")
    embed_provider.set_embedding_model("gemini-embedding-001", 768)
    vec = embed_provider.embed_text("Cardiologist in Cairo")
    assert vec is not None, "Embedding returned None"
    assert len(vec) == 768, f"Expected vector length 768, got {len(vec)}"
    print(f"✓ Embedding generated vector of length {len(vec)}")


def test_generation(gen_provider):
    print("\n=== Testing Gemini Generation ===")
    gen_provider.set_generation_model("gemini-3.5-flash-lite")
    resp = gen_provider.generate_response(
        prompt="Hello, reply with one word: Success"
    )
    assert resp is not None, "generate_response returned None"
    assert resp.message is not None, "response.message is None"
    assert "Success" in (resp.message.content or ""), f"Unexpected content: {resp.message.content}"
    assert resp.prompt_eval_count > 0, "Expected prompt_eval_count > 0"
    print(f"✓ Gemini generation response: {resp.message.content.strip()}")
    print(f"  Token usage: prompt={resp.prompt_eval_count}, completion={resp.eval_count}")


def test_tool_calling(gen_provider):
    print("\n=== Testing Gemini Tool Calling ===")

    def get_clinic_info(city: str) -> str:
        """Get clinic info for a city.
        Args:
            city: City name.
        """
        return f"Clinics open in {city}: Main Health Clinic"

    resp = gen_provider.generate_response(
        prompt="What clinics are open in Cairo?",
        tools=[get_clinic_info],
    )
    assert resp is not None
    assert resp.message is not None
    assert resp.message.tool_calls is not None, "Expected tool_calls to be present"
    assert len(resp.message.tool_calls) > 0, "Expected at least 1 tool call"

    tc = resp.message.tool_calls[0]
    print(f"✓ Tool call received: name={tc.function.name}, args={tc.function.arguments}")
    assert tc.function.name == "get_clinic_info"


def test_agent_integration(gen_provider, embed_provider, config):
    print("\n=== Testing End-to-End Agent with Gemini ===")
    embed_provider.set_embedding_model("gemini-embedding-001", 1024)

    factory_db = VectorDBProviderFactory(config=config)
    client_db = factory_db.create(VectorDBEnums.QDRANT.value)
    if client_db:
        client_db.connect()

    doctor_tool = DoctorTools(
        embedding_client=embed_provider,
        vectordb_client=client_db
    )

    tools = [doctor_tool.search_doctors, web_search]
    agent = Agent(
        client=gen_provider,
        tools=tools,
        max_iterations=5
    )

    query = "Find me a doctor for tooth pain in Dokki"
    answer, messages, cost = agent.run(query)
    print(f"✓ Agent Run finished. Cost: ${cost:.6f}")
    print(f"  Final Response (first 150 chars): {answer[:150]}...")
    assert len(messages) > 1, "Expected messages in history"
    assert answer and len(answer) > 0, "Expected non-empty answer"


if __name__ == "__main__":
    load_dotenv()
    config = get_settings()
    gen_provider, embed_provider = test_factory_and_enums(config)
    test_embedding(embed_provider)
    test_generation(gen_provider)
    test_tool_calling(gen_provider)
    test_agent_integration(gen_provider, embed_provider, config)
    print("\n==========================================")
    print("ALL GEMINI INTEGRATION TESTS PASSED (100%)")
    print("==========================================")

   