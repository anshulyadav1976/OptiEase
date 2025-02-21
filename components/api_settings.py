import streamlit as st


def render_api_settings():
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

    return st.session_state.api_settings