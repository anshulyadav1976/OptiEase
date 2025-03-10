import streamlit as st
from typing import Dict


def render_api_settings() -> Dict:
    """
    Renders the API configuration UI component.
    Returns the current API settings as a dictionary.
    """
    # Initialize session state for API settings if not exists
    if 'api_settings' not in st.session_state:
        st.session_state.api_settings = {
            'provider': 'openai',
            'api_key': '',
            'model': 'gpt-4'
        }

    st.sidebar.header("🔑 API Configuration")

    # Provider selection
    provider = st.sidebar.radio(
        "Select AI Provider",
        ["OpenAI", "Anthropic"],
        index=0 if st.session_state.api_settings['provider'] == 'openai' else 1
    )

    # Convert to lowercase for internal use
    provider_key = provider.lower()

    # API key input with password masking
    api_key = st.sidebar.text_input(
        f"{provider} API Key",
        type="password",
        value=st.session_state.api_settings['api_key'] if st.session_state.api_settings[
                                                              'provider'] == provider_key else "",
        help=f"Your {provider} API key will be used only for this session and won't be stored."
    )

    # Provider-specific model selection
    if provider_key == 'openai':
        model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        default_index = model_options.index(st.session_state.api_settings['model']) if st.session_state.api_settings[
                                                                                           'model'] in model_options else 0
    else:  # anthropic
        model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        default_index = 0

    model = st.sidebar.selectbox(
        f"Select {provider} Model",
        options=model_options,
        index=default_index
    )

    # Update session state if settings changed
    if (provider_key != st.session_state.api_settings['provider'] or
            api_key != st.session_state.api_settings['api_key'] or
            model != st.session_state.api_settings['model']):

        st.session_state.api_settings = {
            'provider': provider_key,
            'api_key': api_key,
            'model': model
        }

        # Show confirmation but don't log the key
        if api_key:
            st.sidebar.success(f"{provider} API configured successfully!")

    # Add security notes
    with st.sidebar.expander("📋 Security Information"):
        st.markdown("""
        ### Security Notes
        - Your API key is only stored in your browser's session state
        - Keys are never logged or stored on servers
        - Session data is cleared when you close the browser
        - All data processing happens locally in your browser
        - We don't store your uploaded data files
        """)

    with st.sidebar.expander("🔑 API Settings", expanded=False):
        st.markdown("""
        ### Model Selection
        Choose the AI model that best fits your needs:
        
        #### Cost-Effective Options (Long Context)
        - Fast responses, good for basic tasks
        - Lower cost per token
        - Still capable of handling complex problems
        """)
        
        cost_effective_model = st.selectbox(
            "Cost-Effective Model",
            [
                "gpt-3.5-turbo-0125 (OpenAI)",  # $0.0005/1K input, $0.0015/1K output, 16K context
                "claude-3-haiku (Anthropic)",    # $0.00025/1K input, $0.00125/1K output, 200K context
                "claude-3.5-haiku (Anthropic)",  # $0.0008/1K input, $0.004/1K output, 200K context
            ],
            help="Choose a cost-effective model for basic tasks and quick responses"
        )
        
        st.markdown("""
        #### High-Capability Options
        - Superior reasoning and analysis
        - Better at complex tasks
        - More nuanced responses
        - Higher cost per token
        """)
        
        high_capability_model = st.selectbox(
            "High-Capability Model",
            [
                "gpt-4-turbo-0125 (OpenAI)",     # $0.01/1K input, $0.03/1K output, 128K context
                "claude-3-opus (Anthropic)",      # $0.015/1K input, $0.075/1K output, 200K context
                "claude-3-sonnet (Anthropic)",    # $0.003/1K input, $0.015/1K output, 200K context
                "claude-3.5-sonnet (Anthropic)",  # $0.003/1K input, $0.015/1K output, 200K context
            ],
            help="Choose a high-capability model for complex analysis and reasoning"
        )
        
        st.markdown("### API Keys")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for OpenAI models (GPT-3.5, GPT-4)"
        )
        
        anthropic_api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Required for Anthropic models (Claude)"
        )
        
        st.markdown("""
        ### Model Usage Guidelines
        
        #### When to use Cost-Effective Models:
        - Quick responses needed
        - Basic analysis and explanations
        - High-volume tasks
        - Budget constraints
        - Simple problem-solving
        
        #### When to use High-Capability Models:
        - Complex analysis required
        - Detailed technical explanations
        - Critical decision support
        - Advanced problem-solving
        - Nuanced understanding needed
        
        #### Context Windows:
        - GPT-3.5: 16K tokens
        - GPT-4 Turbo: 128K tokens
        - Claude 3/3.5: 200K tokens
        """)
        
        return {
            "cost_effective_model": cost_effective_model,
            "high_capability_model": high_capability_model,
            "openai_api_key": openai_api_key,
            "anthropic_api_key": anthropic_api_key
        }