from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from groq_news_server import (
    RequestError,
    build_messages,
    build_groq_request,
    DEFAULT_GROQ_MODEL,
    describe_provider_error,
    generate_release,
    groq_is_configured,
    parse_generated_release,
    validate_case_payload,
)


def valid_payload():
    return {
        "case_id": "12345678-1234-1234-1234-123456789abc",
        "case_number": "DELO-0001",
        "case_name": "О деле",
        "suspect_name": "Гражданин Тестов",
        "charge": "Нарушение порядка",
        "article": "Статья 1",
        "incident": "Событие по материалам дела.",
        "verdict": "Штраф",
        "verdict_details": "Штраф",
        "world_tone": "неопределённое",
    }


class NewsPayloadTests(unittest.TestCase):
    def test_groq_configuration_requires_non_empty_key(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "  "}, clear=True):
            self.assertFalse(groq_is_configured())
        with patch.dict(os.environ, {"GROQ_API_KEY": "configured"}, clear=True):
            self.assertTrue(groq_is_configured())

    def test_provider_errors_include_useful_details_without_exposing_key(self):
        with patch.dict(
            os.environ, {"GROQ_API_KEY": "test-secret"}, clear=True
        ):
            details = describe_provider_error(
                401,
                b'{"error":{"code":"invalid_api_key","message":"Bearer test-secret rejected"}}',
            )
        self.assertIn("HTTP 401", details)
        self.assertIn("invalid_api_key", details)
        self.assertNotIn("test-secret", details)

    def test_valid_payload_is_normalized(self):
        payload = valid_payload()
        payload["suspect_name"] = "  Гражданин Тестов  "
        self.assertEqual(
            validate_case_payload(payload)["suspect_name"], "Гражданин Тестов"
        )

    def test_rejects_missing_case_facts(self):
        payload = valid_payload()
        del payload["incident"]
        with self.assertRaises(RequestError):
            validate_case_payload(payload)

    def test_rejects_invalid_world_tone(self):
        payload = valid_payload()
        payload["world_tone"] = "90"
        with self.assertRaises(RequestError):
            validate_case_payload(payload)

    def test_case_facts_are_data_not_instructions(self):
        messages = build_messages(validate_case_payload(valid_payload()))
        self.assertIn("untrusted data", messages[0]["content"])
        self.assertIn("fictional", messages[0]["content"])
        self.assertIn("in natural Russian", messages[0]["content"])
        self.assertIn("Russian-language", messages[0]["content"])
        self.assertIn("case facts only, not", messages[1]["content"])

    def test_groq_request_uses_proxy_compatible_user_agent(self):
        request = build_groq_request(
            "test-key", DEFAULT_GROQ_MODEL, validate_case_payload(valid_payload())
        )
        self.assertEqual(
            request.get_header("User-agent"), "WorldNewsBackend/1.0"
        )
        request_payload = json.loads(request.data)
        self.assertEqual(request_payload["model"], DEFAULT_GROQ_MODEL)
        self.assertEqual(
            request_payload["response_format"], {"type": "json_object"}
        )
        self.assertEqual(request_payload["max_tokens"], 500)
        messages = request_payload["messages"]
        self.assertIn("JSON object", messages[0]["content"])
        self.assertIn("headline", messages[0]["content"])
        self.assertIn("script", messages[0]["content"])

    def test_retry_request_uses_strict_prompt_without_json_mode(self):
        request = build_groq_request(
            "test-key",
            DEFAULT_GROQ_MODEL,
            validate_case_payload(valid_payload()),
            retry_without_json_mode=True,
        )
        request_payload = json.loads(request.data)
        self.assertNotIn("response_format", request_payload)
        self.assertIn("exactly one valid JSON object",
            request_payload["messages"][0]["content"])
        self.assertIn("in natural Russian",
            request_payload["messages"][0]["content"])
        self.assertEqual(request_payload["temperature"], 0.2)

    def test_retries_json_validation_failure_without_json_mode(self):
        invalid_json_error = HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            400,
            "Bad Request",
            {},
            BytesIO(
                b'{"error":{"code":"json_validate_failed",'
                b'"message":"Failed to generate JSON"}}'
            ),
        )
        successful_response = BytesIO(json.dumps({
            "choices": [{
                "message": {
                    "content": (
                        '{"headline":"A headline",'
                        '"script":"A short news report."}'
                    )
                }
            }]
        }).encode("utf-8"))
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test-key", "GROQ_MODEL": DEFAULT_GROQ_MODEL},
            clear=True,
        ):
            with patch(
                "groq_news_server.urllib.request.urlopen",
                side_effect=[invalid_json_error, successful_response],
            ) as urlopen:
                release = generate_release(
                    validate_case_payload(valid_payload())
                )

        self.assertEqual(
            release,
            {"headline": "A headline", "script": "A short news report."},
        )
        self.assertEqual(urlopen.call_count, 2)
        retry_request = urlopen.call_args_list[1].args[0]
        self.assertNotIn("response_format", json.loads(retry_request.data))

    def test_release_parser_accepts_json_fenced_text_from_retry(self):
        release = parse_generated_release({
            "choices": [{
                "message": {
                    "content": (
                        '```json\n{"headline":"A headline",'
                        '"script":"A short news report."}\n```'
                    )
                }
            }]
        })
        self.assertEqual(release["headline"], "A headline")

    def test_release_parser_extracts_json_surrounded_by_explanatory_text(self):
        release = parse_generated_release({
            "choices": [{
                "message": {
                    "content": (
                        'Here is the requested result:\n'
                        '{"headline":"A headline",'
                        '"script":"A short news report."}\n'
                        "I hope this helps."
                    )
                }
            }]
        })
        self.assertEqual(release["script"], "A short news report.")

    def test_retries_successful_provider_response_with_invalid_release(self):
        malformed_response = BytesIO(json.dumps({
            "choices": [{
                "message": {"content": "I cannot provide valid JSON."}
            }]
        }).encode("utf-8"))
        successful_response = BytesIO(json.dumps({
            "choices": [{
                "message": {
                    "content": (
                        '{"headline":"A headline",'
                        '"script":"A short news report."}'
                    )
                }
            }]
        }).encode("utf-8"))
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test-key", "GROQ_MODEL": DEFAULT_GROQ_MODEL},
            clear=True,
        ):
            with patch(
                "groq_news_server.urllib.request.urlopen",
                side_effect=[malformed_response, successful_response],
            ) as urlopen:
                release = generate_release(
                    validate_case_payload(valid_payload())
                )

        self.assertEqual(release["headline"], "A headline")
        self.assertEqual(urlopen.call_count, 2)
        retry_request = urlopen.call_args_list[1].args[0]
        self.assertNotIn("response_format", json.loads(retry_request.data))

if __name__ == "__main__":
    unittest.main()
