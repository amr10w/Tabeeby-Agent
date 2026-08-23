import json
import logging
from typing import Any, Dict, List, Optional
from tavily import TavilyClient

logger = logging.getLogger(__name__)

def web_search(query:str,max_results:int=3) -> str:
    """Search the public web for general medical info, conditions, or drug interactions.

    Use when the patient asks about rare symptoms, specific medications,
    or medical conditions not covered by the doctor catalog.

    Args:
        query: Clear search keywords (e.g., 'Metformin side effects', 'Lyme disease rash').
        max_results: Maximum number of search snippets to return (default: 3).

    Returns:
        JSON string containing snippets with title, content body, and URL.
    """

    if not query or not str(query).strip():
        return json.dumps({"error": "Empty search query provided."})

    try:
        client = TavilyClient()

        response = client.search(
            query=str(query).strip(),
            max_results=max_results,
            search_depth="basic",
        )

        raw_results = response.get("results", [])

        if not raw_results:
            return json.dumps({"message": "No web results found for the given query."})

        compact_results: List[Dict[str, Any]] = [
            {
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "url": item.get("url", ""),
            }
            for item in raw_results
        ]

        return json.dumps(compact_results, ensure_ascii=False)

    

    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        # Always return error as text data so the LLM reasoning loop stays alive
        return json.dumps({"error": f"Failed to execute web search: {str(e)}"})