import unittest

from personal_assistant.skill_references import SkillReferenceError, parse_skill_references


class SkillReferenceTests(unittest.TestCase):
    names = ("calendar", "email", "web-research")

    def test_multiple_references_preserve_order_and_deduplicate(self):
        self.assertEqual(parse_skill_references(
            "Use @email, then @calendar. Also @email and @web-research!", self.names
        ), ("email", "calendar", "web-research"))

    def test_literals_are_not_references(self):
        for text in (
            "me@email.com", "https://host/@calendar", r"Use \@missing literally",
            "`@missing` @email", "``literal ` @missing`` @email",
            "```python\n@missing\n```\n@email", "~~~\n@missing\n~~~\n@email",
            "```\n@missing", "`@missing", "@email.com", "hello@calendar",
            "`literal`@missing", "``literal``@missing",
        ):
            with self.subTest(text=text):
                expected = ("email",) if text.endswith(" @email") or text.endswith("\n@email") else ()
                self.assertEqual(parse_skill_references(text, self.names), expected)

    def test_unknown_reference_has_actionable_error(self):
        with self.assertRaisesRegex(SkillReferenceError, r"@missing.*\/skills"):
            parse_skill_references("@email @missing", self.names)

    def test_bare_marker_requires_a_selection(self):
        with self.assertRaises(SkillReferenceError):
            parse_skill_references("Use @", self.names)

    def test_fence_must_close_with_matching_marker(self):
        self.assertEqual(parse_skill_references("````\n```\n@missing\n````\n@email", self.names), ("email",))


if __name__ == "__main__":
    unittest.main()
