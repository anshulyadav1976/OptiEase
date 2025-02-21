import openai
import anthropic
from typing import Dict, Any, Optional


def get_explanation(context: str, analysis_type: str, api_settings: Dict[str, Any]) -> str:
    """
    Get natural language explanation from selected AI provider

    Args:
        context: The analysis context or data to explain
        analysis_type: Type of analysis being explained
        api_settings: Dictionary containing API configuration

    Returns:
        str: The explanation text
    """
    if not api_settings['api_key']:
        return "Please configure your API key in the settings to get explanations."

    try:
        if api_settings['provider'] == 'openai':
            return _get_openai_explanation(context, analysis_type, api_settings)
        elif api_settings['provider'] == 'anthropic':
            return _get_anthropic_explanation(context, analysis_type, api_settings)
        else:
            return f"Unsupported provider: {api_settings['provider']}"
    except Exception as e:
        return f"Explanation service error: {str(e)}"


def _get_openai_explanation(context: str, analysis_type: str, api_settings: Dict[str, Any]) -> str:
    """Internal function to handle OpenAI API requests"""
    client = openai.OpenAI(api_key=api_settings['api_key'])
    response = client.chat.completions.create(
        model=api_settings['model'],
        messages=[
            {"role": "system", "content": "You are an expert in operational research and statistics."},
            {"role": "user",
             "content": f"Explain the following {analysis_type} in simple terms for a business user: {context}"}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content


def _get_anthropic_explanation(context: str, analysis_type: str, api_settings: Dict[str, Any]) -> str:
    """Internal function to handle Anthropic API requests"""
    client = anthropic.Anthropic(api_key=api_settings['api_key'])
    response = client.messages.create(
        model=api_settings['model'],
        max_tokens=500,
        system="You are an expert in operational research and statistics.",
        messages=[
            {"role": "user",
             "content": f"Explain the following {analysis_type} in simple terms for a business user: {context}"}
        ]
    )
    return response.content[0].text