from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING
import logging
import re
from schema import Schema, Literal

from griptape.events import EventBus
from griptape.artifacts import ErrorArtifact, TextArtifact
from griptape.drivers import GriptapeCloudRulesetDriver
from griptape.rules import Ruleset, Rule, JsonSchemaRule
from griptape.structures import Agent
from griptape.memory.structure import ConversationMemory, Run

from griptape_structure_slack_handler.griptape_event_handlers import ToolEvent

from .griptape_tool_box import get_tools
from .griptape_config import load_griptape_config, set_thread_alias
from .features import dynamic_rulesets_enabled, dynamic_tools_enabled

if TYPE_CHECKING:
    from griptape.events import EventListener


logger = logging.getLogger()


load_griptape_config()


def try_add_to_thread(
    message: str, *, thread_alias: Optional[str] = None, user_id: str
) -> None:
    set_thread_alias(thread_alias)
    # find all the user_ids @ mentions in the message
    mentioned_user_ids = re.findall(r"<@([\w]+)>", message)
    rulesets = [Ruleset(name=mentioned_user) for mentioned_user in mentioned_user_ids]
    for ruleset in rulesets:
        # If the message is mentioning the bot, don't add it to the memory
        # because the bot will already be responding to the message,
        # and the message will be in conversation memory already
        if ruleset.meta.get("type") == "bot":
            return

    memory = ConversationMemory()
    # WIP. since messages that do not tag the bot are not being added to the cloud Thread,
    # the bot can miss context. This inserts those messages into the Thread, which
    # later can be used to provide context via ConversationMemory. this seems to work okay,
    # but it can confuse the LLM
    memory.add_run(
        Run(
            input=TextArtifact(
                f"Do not respond. Only use this message for future context. Message: 'user {user_id}: {message}'"
            ),
            output=TextArtifact(""),
        )
    )


def get_rulesets(**kwargs) -> list[Ruleset]:
    rulesets = (
        [Ruleset(name=value) for value in kwargs.values()]
        if dynamic_rulesets_enabled()
        else []
    )
    rulesets.extend(_get_default_rulesets())
    return rulesets


def _get_default_rulesets() -> list[Ruleset]:
    return [
        # Knowledge Base Ruleset
        Ruleset(
            ruleset_driver=GriptapeCloudRulesetDriver(
                raise_not_found=True, ruleset_id="f5a9c72b-b367-403e-9872-cdd85431e898"
            ),
        ),
        # Slack Personality Ruleset
        Ruleset(
            ruleset_driver=GriptapeCloudRulesetDriver(
                raise_not_found=True, ruleset_id="60a368ef-f5ac-4c63-990c-80a7364a22a0"
            ),
        ),
        # Zach Prime ID Ruleset
        Ruleset(
            ruleset_driver=GriptapeCloudRulesetDriver(
                raise_not_found=True, ruleset_id="6f35283b-e336-433b-84dd-9ee53e8c69bf"
            ),
        ),
        # Slack Formatting Ruleset
        Ruleset(
            ruleset_driver=GriptapeCloudRulesetDriver(
                raise_not_found=True, ruleset_id="b2cae474-a25c-476b-90a1-f39d588dd711"
            ),
        ),
        # Slack Tool Ruleset
        Ruleset(
            ruleset_driver=GriptapeCloudRulesetDriver(
                raise_not_found=True, ruleset_id="31b9d849-dade-4c3f-ac31-377ff6f02307"
            ),
        ),
    ]


def agent(
    message: str,
    *,
    thread_alias: Optional[str] = None,
    user_id: str,
    rulesets: list[Ruleset],
    event_listeners: list[EventListener],
    stream: bool,
) -> str:
    set_thread_alias(thread_alias)
    dynamic_tools = dynamic_tools_enabled() or any(
        [ruleset.meta.get("dynamic_tools", False) for ruleset in rulesets]
    )
    tools = get_tools(message, dynamic=dynamic_tools)
    EventBus.add_event_listeners(event_listeners)

    if dynamic_tools:
        EventBus.publish_event(ToolEvent(tools=tools, stream=stream), flush=True)

    agent = Agent(
        input="user_id '<@{{ args[0] }}>': {{ args[1] }}",
        tools=tools,
        rulesets=rulesets,
        stream=stream,
    )
    output = agent.run(user_id, message).output
    if isinstance(output, ErrorArtifact):
        raise ValueError(output.to_text())
    return output.to_text()


def is_relevant_response(message: str, response: str) -> bool:
    agent = Agent(
        input="Given the following message: '{{ args[0] }}', is the following response helpful and relevant? Response: {{ args[1] }}",
        rulesets=[
            Ruleset(
                rules=[
                    Rule(
                        "You should respond if the response is helpful and relevant to the user"
                    ),
                    Rule(
                        "If the message is a question, the response should be shown to the user if the response is helpful and relevant."
                    ),
                    JsonSchemaRule(
                        Schema(
                            {
                                Literal(
                                    "should_respond",
                                    description="Boolean value that determines if the given agent response should be sent to the user.",
                                ): bool
                            }
                        ).json_schema("should_respond")
                    ),
                ]
            )
        ],
        stream=False,
    )

    output = agent.run(message, response).output
    if isinstance(output, ErrorArtifact):
        raise ValueError(output.to_text())
    return json.loads(output.to_text())["should_respond"]
