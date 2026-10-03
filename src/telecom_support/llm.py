"""Shared structured Groq calls with safe errors and no automatic paid retries."""
import os
from dotenv import load_dotenv
from groq import Groq, APIStatusError, APIConnectionError
from pydantic import ValidationError

class ServiceError(RuntimeError):
    pass

class StructuredLLM:
    def __init__(self, root):
        load_dotenv(root / '.env', override=False)
        key = os.getenv('GROQ_API_KEY', '').strip()
        if not key:
            raise ServiceError('Set GROQ_API_KEY in your local .env file.')
        self.model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
        self.client = Groq(api_key=key, timeout=60, max_retries=0)

    def call(self, name, schema, messages, max_tokens=2048):
        try:
            response = self.client.chat.completions.create(model=self.model, temperature=0,
                max_completion_tokens=max_tokens, messages=messages,
                response_format={'type':'json_schema', 'json_schema': {
                    'name':name, 'strict':True, 'schema':schema.model_json_schema()}})
            if not response.choices or response.choices[0].finish_reason != 'stop':
                raise ServiceError('LLM response was incomplete. Try again later.')
            return schema.model_validate_json(response.choices[0].message.content or '')
        except APIStatusError as exc:
            if exc.status_code == 429:
                retry = exc.response.headers.get('retry-after', 'not supplied')
                raise ServiceError(f'Groq rate limit reached. Retry-after seconds: {retry}. Check your quota.') from None
            raise ServiceError(f'Groq HTTP {exc.status_code}; check key/model access.') from None
        except APIConnectionError:
            raise ServiceError('Groq connection failed or timed out. Check connectivity.') from None
        except ValidationError:
            raise ServiceError('LLM output failed schema validation.') from None

    def close(self):
        self.client.close()
