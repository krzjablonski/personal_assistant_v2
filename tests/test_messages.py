import json
import unittest

from llm.messages import Image, Message, ToolCall


class TestMessages(unittest.TestCase):
    def test_plain_text_has_no_content_wrapper(self):
        message = Message(role="user", text="Hello")
        self.assertEqual(message.to_dict(), {"role": "user", "text": "Hello"})
        self.assertEqual(Message.from_dict(message.to_dict()), message)

    def test_all_fields_survive_json_round_trip(self):
        messages = [
            Message("user", "Look", images=[Image("aGVsbG8=", "image/png")]),
            Message("assistant", "Checking", tool_calls=[ToolCall("c1", "lookup", {"q": "x"})],
                    reasoning="Plan", provider_data={"vendor": {"signature": "opaque"}}),
            Message("tool", "Unavailable", tool_call_id="c1", tool_name="lookup", is_error=True),
        ]
        for message in messages:
            with self.subTest(role=message.role):
                saved = json.loads(json.dumps(message.to_dict()))
                self.assertEqual(Message.from_dict(saved), message)

    def test_unknown_fields_fail_instead_of_silently_losing_data(self):
        with self.assertRaises(TypeError):
            Message.from_dict({"role": "user", "text": "hello", "unknown": 1})

    def test_system_instructions_use_the_separate_client_argument(self):
        with self.assertRaises(ValueError):
            Message.from_dict({"role": "system", "text": "instructions"})

    def test_message_lists_are_independent(self):
        one, two = Message("assistant"), Message("assistant")
        one.tool_calls.append(ToolCall("c", "lookup", {}))
        self.assertEqual(two.tool_calls, [])
