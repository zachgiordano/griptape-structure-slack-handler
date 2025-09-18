import os
from typing import Optional

from griptape.configs import Defaults
from griptape.configs.drivers import AzureOpenAiDriversConfig
from griptape.drivers import (
    GriptapeCloudConversationMemoryDriver,
    GriptapeCloudRulesetDriver,
)


def load_griptape_config() -> None:
    """Load the Default Griptape configuration. If no OPENAI_API_KEY is found, use Azure OpenAI drivers."""
    if "OPENAI_API_KEY" not in os.environ:
        Defaults.drivers_config = AzureOpenAiDriversConfig(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )

    Defaults.drivers_config.ruleset_driver = GriptapeCloudRulesetDriver(raise_not_found=False)
    Defaults.drivers_config.conversation_memory_driver = GriptapeCloudConversationMemoryDriver()


def set_thread_alias(thread_alias: Optional[str]) -> None:
    """Set the thread alias for the conversation memory driver."""
    Defaults.drivers_config.conversation_memory_driver.alias = thread_alias
