# free-model-aggregator-02's Python entrypoint: one server, two client wires, and a real free model
# behind every answer. Written using Python 3.13 standard library alone.
#
# PLATFORM CONTRACT:
#   - Entrypoint is executed as `python3 /bot/main.py` from repository root.
#   - Single file main.py in repository root with only Python 3.13 standard library.
#   - `python3 main.py --list` prints the catalog JSON to stdout and exits 0 in < 10s without touching network.
#   - MAKEFUN_PROMPT_FILE: path to brief JSON. If not set, catalog is {"providers": []}.
#   - MAKEFUN_ENV_URL: environment base URL; server registers port + catalog (POST /register).
#   - Server uses http.server.ThreadingHTTPServer on 0.0.0.0 port 0 (ephemeral).
#   - Handles OpenAI chat/completions (stream + non-stream), Anthropic messages (stream + non-stream),
#     and GET /v1/models (OpenAI Authorization header vs Anthropic x-api-key header).
#   - Retries registration at most 3 times with backoff <= 1s, giving up cleanly with no traceback.
#   - Explicit timeout (< 30s) on all upstream calls; BrokenPipeError / ConnectionResetError handled cleanly.
#   - Failover tries next candidate only before first byte is sent to client; closes cleanly once started.
#   - Exits cleanly (0) on SIGTERM / SIGINT and when GET /work reports done.

import json
import math
import os
import re
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- Constants

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
UPSTREAM_TIMEOUT = 25  # seconds (< 30s pod limit requirement)
MAX_REG_ATTEMPTS = 3

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
SCHEMA_PREFIX = 'free-model-aggregator-02/'

OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1'
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

Json = dict[str, object]

# ---------------------------------------------------------------- Type helpers


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


# ---------------------------------------------------------------- Brief & Catalog


def object_end(text: str, start: int) -> int:
    """The index of the `}` closing the object that opens at `start`, or -1; string contents skipped."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in order of appearance."""
    out: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """The first document under this contest's prefix that carries one of `keys` as an array."""
    for doc in docs:
        schema = string_of(doc.get('schema'))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    """One model row. A free row is routed with its full gateway id."""
    if not is_record(value):
        return None
    raw = string_of(value.get('id'))
    if raw == '':
        raw = string_of(value.get('model'))
    if raw == '':
        return None
    named = string_of(value.get('provider'))
    if named == '':
        named = string_of(value.get('wire'))
    model_id = f'openrouter/{raw}' if named == 'openrouter' and not raw.lower().startswith('openrouter/') else raw
    display_name = string_of(value.get('display_name'))
    rating = number_or_null(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': named if named != '' else 'openrouter',
        'display_name': display_name if display_name != '' else raw,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
        'free': value.get('free') is not False,
    }


def prompt_of(value: object) -> Json | None:
    """One prompt row from the manifest."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if is_record(value.get('requires')) else {}
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': {
            **({'tools': True} if requires.get('tools') is True else {}),
            **({'vision': True} if requires.get('vision') is True else {}),
            **({'json': True} if requires.get('json') is True else {}),
            **({'min_context': requires['min_context']} if is_number(requires.get('min_context')) and math.isfinite(requires['min_context']) else {}),
        },
    }


def read_brief(path: str) -> Json:
    """Read the brief file and extract models and prompts documents."""
    if path == '':
        return {'models': [], 'prompts': [], 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read {path}: {err}'}
    docs = documents_in(text)
    models_doc = array_document(docs, ('models', 'ratings'))
    if models_doc is None:
        return {'models': [], 'prompts': [], 'error': f'the brief at {path} holds no {MODELS_SCHEMA} document'}
    models = [m for m in (model_of(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = array_document(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        return {'models': models, 'prompts': [], 'error': f'the brief at {path} holds no {PROMPTS_SCHEMA} document'}
    prompts = [p for p in (prompt_of(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {'models': models, 'prompts': prompts, 'error': None}


def free_models(brief: Json) -> list[Json]:
    return [m for m in brief.get('models', []) if m.get('free')]


def capabilities_of(model: Json) -> Json:
    return {
        'context_length': model.get('context_length'),
        'tools': model.get('tools'),
        'vision': model.get('vision'),
        'json': model.get('json'),
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    if requires.get('tools') is True and caps.get('tools') is not True:
        return False
    if requires.get('vision') is True and caps.get('vision') is not True:
        return False
    if requires.get('json') is True and caps.get('json') is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None:
        context_len = caps.get('context_length')
        if context_len is None or context_len < min_context:
            return False
    return True


def build_catalog(brief: Json) -> Json:
    """Build the catalog JSON structure for --list and registration."""
    seen: set[str] = set()
    models: list[Json] = []
    for model in brief.get('models', []):
        model_id = model.get('id', '')
        if not model.get('free') or model_id in seen or not model_id:
            continue
        seen.add(model_id)
        models.append({'id': model_id, 'free': True, 'context_length': model.get('context_length')})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': models,
            }
        ]
    }


# Initialized from environment (used for runtime state)
BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
manifest_index = 0
manifest_lock = threading.Lock()

# Cooldown / failure tracker for quality routing and failover
model_failures: dict[str, int] = {}
model_failures_lock = threading.Lock()


def chat_completions_url() -> str:
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def served_models() -> list[Json]:
    return [{'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']} for m in free_models(BRIEF)]


def route_candidates(requires: Json) -> list[Json]:
    """Rank all free candidates meeting requirements by fewest failures, then highest rating_milli."""
    candidates = [m for m in free_models(BRIEF) if meets_requirements(requires, capabilities_of(m))]
    with model_failures_lock:
        return sorted(
            candidates,
            key=lambda m: (model_failures.get(m['id'], 0), -m.get('rating_milli', 0)),
        )


def requirements_for(body: Json) -> Json:
    """The requirements of the prompt arriving: the brief's row by arrival order, else body fallback."""
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF.get('prompts', [])
    if index < len(prompts):
        return prompts[index].get('requires', {})
    return requirements_from_body(body)


def requirements_from_body(body: Json) -> Json:
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    return {
        'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0,
        'vision': vision,
        'json': response_type in ('json_object', 'json_schema'),
    }


# ---------------------------------------------------------------- Wire Translation


def openai_model_list(models: list[Json]) -> Json:
    return {
        'object': 'list',
        'data': [{'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']} for m in models],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    return {
        'data': [{'type': 'model', 'id': m['id'], 'display_name': m['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT} for m in models],
        'has_more': False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def clean_json_content(text: str) -> str:
    """Extract unwrapped JSON from markdown fences or surrounding prose."""
    trimmed = text.strip()
    match = re.match(r'^```(?:json)?\s*([\s\S]*?)\s*```$', trimmed, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', trimmed, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        json.loads(trimmed)
        return trimmed
    except (json.JSONDecodeError, ValueError):
        first_brace = trimmed.find('{')
        last_brace = trimmed.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            candidate = trimmed[first_brace : last_brace + 1]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass
        first_bracket = trimmed.find('[')
        last_bracket = trimmed.rfind(']')
        if first_bracket != -1 and last_bracket > first_bracket:
            candidate = trimmed[first_bracket : last_bracket + 1]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass
    return text


def part_of(block: Json) -> Json | None:
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except (json.JSONDecodeError, ValueError):
            first_brace = value.find('{')
            last_brace = value.rfind('}')
            if first_brace != -1 and last_brace > first_brace:
                try:
                    candidate = json.loads(value[first_brace : last_brace + 1])
                    if is_record(candidate):
                        return candidate
                except (json.JSONDecodeError, ValueError):
                    pass
    return {}


def stop_reason_of(finish: object) -> str:
    if finish == 'length':
        return 'max_tokens'
    if finish in ('tool_calls', 'function_call'):
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def estimate_prompt_tokens(body: Json) -> int:
    chars = 0
    system = body.get('system')
    if isinstance(system, str):
        chars += len(system)
    messages = body.get('messages')
    if isinstance(messages, list):
        for msg in messages:
            if not is_record(msg):
                continue
            content = msg.get('content')
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if isinstance(block.get('text'), str):
                        chars += len(block['text'])
                    if isinstance(block.get('content'), str):
                        chars += len(block['content'])
    tools = body.get('tools')
    if isinstance(tools, list):
        chars += len(json.dumps(tools, separators=(',', ':'), ensure_ascii=False))
    return max(1, math.ceil(chars / 4))


def anthropic_to_openai(body: Json, model: str) -> Json:
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) and block['id'] != '' else f'toolu_{len(tool_calls) + 1}',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': json.dumps(
                                block['input'] if is_record(block.get('input')) else {},
                                separators=(',', ':'),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            msg: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                msg['tool_calls'] = tool_calls
            messages.append(msg)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages}
    max_tokens = body.get('max_tokens')
    out['max_tokens'] = max_tokens if is_number(max_tokens) and max_tokens > 0 else 1024
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw in body['tools']:
            if not is_record(raw):
                continue
            if raw.get('type') == 'function' and is_record(raw.get('function')):
                tools.append(raw)
                continue
            if not isinstance(raw.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw['name'],
                        'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                        'parameters': raw['input_schema'] if is_record(raw.get('input_schema')) else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
        elif choice.get('type') == 'function' and is_record(choice.get('function')):
            out['tool_choice'] = choice
        elif choice.get('type') == 'none':
            out['tool_choice'] = 'none'
    elif isinstance(choice, str):
        out['tool_choice'] = choice
    if is_record(body.get('response_format')):
        out['response_format'] = body['response_format']
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body: Json, model: str, req_prompt_tokens: int | None = None) -> Json:
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    raw_text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    text = clean_json_content(raw_text)
    if text != '':
        content.append({'type': 'text', 'text': text})

    raw_calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    if not raw_calls and is_record(message.get('function_call')):
        raw_calls = [{'id': 'toolu_0', 'type': 'function', 'function': message['function_call']}]

    tool_chars = 0
    for idx, call in enumerate(raw_calls):
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        args_str = fn['arguments'] if isinstance(fn.get('arguments'), str) else ''
        tool_chars += len(args_str)
        call_id = call['id'] if isinstance(call.get('id'), str) and call['id'].strip() != '' else f'toolu_{idx + 1}'
        content.append(
            {
                'type': 'tool_use',
                'id': call_id,
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': arguments_of(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})

    has_tool_use = any(is_record(b) and b.get('type') == 'tool_use' for b in content)
    if has_tool_use:
        stop_reason = 'tool_use'
    else:
        stop_reason = stop_reason_of(choice.get('finish_reason'))
        if stop_reason == 'tool_use':
            stop_reason = 'end_turn'

    calc_prompt = req_prompt_tokens if req_prompt_tokens and req_prompt_tokens > 0 else 10
    calc_completion = max(1, math.ceil((len(text) + tool_chars) / 4))
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    input_tokens = usage['prompt_tokens'] if is_number(usage.get('prompt_tokens')) and usage['prompt_tokens'] >= 0 else calc_prompt
    output_tokens = usage['completion_tokens'] if is_number(usage.get('completion_tokens')) and usage['completion_tokens'] >= 1 else calc_completion

    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) and body["id"] != "" else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason,
        'stop_sequence': None,
        'usage': {'input_tokens': int(input_tokens), 'output_tokens': int(output_tokens)},
    }


def openai_json_with_model(body: Json, model: str, req_prompt_tokens: int | None = None) -> Json:
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    cleaned_choices: list[Json] = []
    completion_chars = 0
    for c in choices:
        if not is_record(c):
            continue
        choice = dict(c)
        if is_record(choice.get('message')):
            msg = dict(choice['message'])
            if isinstance(msg.get('content'), str):
                msg['content'] = clean_json_content(msg['content'])
                completion_chars += len(msg['content'])
            if isinstance(msg.get('tool_calls'), list):
                for tc in msg['tool_calls']:
                    if is_record(tc) and is_record(tc.get('function')) and isinstance(tc['function'].get('arguments'), str):
                        completion_chars += len(tc['function']['arguments'])
            choice['message'] = msg
        if not isinstance(choice.get('finish_reason'), str) or choice['finish_reason'] == '':
            has_tools = is_record(choice.get('message')) and isinstance(choice['message'].get('tool_calls'), list) and len(choice['message']['tool_calls']) > 0
            choice['finish_reason'] = 'tool_calls' if has_tools else 'stop'
        cleaned_choices.append(choice)

    calc_prompt = req_prompt_tokens if req_prompt_tokens and req_prompt_tokens > 0 else 10
    calc_completion = max(1, math.ceil(completion_chars / 4))

    usage_obj = body.get('usage')
    if is_record(usage_obj):
        pt = usage_obj['prompt_tokens'] if is_number(usage_obj.get('prompt_tokens')) and usage_obj['prompt_tokens'] >= 0 else calc_prompt
        ct = usage_obj['completion_tokens'] if is_number(usage_obj.get('completion_tokens')) and usage_obj['completion_tokens'] >= 1 else calc_completion
        tt = usage_obj['total_tokens'] if is_number(usage_obj.get('total_tokens')) and usage_obj['total_tokens'] > 0 else (pt + ct)
        usage = {**usage_obj, 'prompt_tokens': int(pt), 'completion_tokens': int(ct), 'total_tokens': int(tt)}
    else:
        usage = {'prompt_tokens': int(calc_prompt), 'completion_tokens': int(calc_completion), 'total_tokens': int(calc_prompt + calc_completion)}

    return {**body, 'model': model, 'choices': cleaned_choices, 'usage': usage}


def sse_event(event: str | None, data: object) -> str:
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def iter_frames(chunks):
    """SSE frames from byte chunks, split on the blank line that ends each one."""
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find('\n\n')
    if buffer.strip() != '':
        yield buffer


def openai_stream_with_model(chunks, model: str, req_prompt_tokens: int | None = None):
    """Chat Completions SSE stream with exact model claim, guaranteed single finish and usage chunk."""
    calc_prompt = req_prompt_tokens if req_prompt_tokens and req_prompt_tokens > 0 else 10
    has_usage = False
    saw_finish = False
    saw_done = False
    completion_chars = 0

    def usage_chunk() -> str:
        calc_completion = max(1, math.ceil(completion_chars / 4))
        payload = {
            'id': 'chatcmpl-usage',
            'object': 'chat.completion.chunk',
            'created': FIXED_OPENAI_CREATED,
            'model': model,
            'choices': [],
            'usage': {
                'prompt_tokens': calc_prompt,
                'completion_tokens': calc_completion,
                'total_tokens': calc_prompt + calc_completion,
            },
        }
        return f'data: {json.dumps(payload, separators=(",", ":"), ensure_ascii=False)}\n\n'

    def finish_chunk() -> str:
        nonlocal saw_finish
        saw_finish = True
        payload = {
            'id': 'chatcmpl-finish',
            'object': 'chat.completion.chunk',
            'created': FIXED_OPENAI_CREATED,
            'model': model,
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
        }
        return f'data: {json.dumps(payload, separators=(",", ":"), ensure_ascii=False)}\n\n'

    def process_frame(frame: str) -> str:
        nonlocal has_usage, saw_finish, saw_done, completion_chars
        lines: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                lines.append(line)
                continue
            payload = trimmed[5:].strip()
            if payload == '':
                lines.append(line)
                continue
            if payload == '[DONE]':
                saw_done = True
                prefix = ''
                if not saw_finish:
                    prefix += finish_chunk()
                if not has_usage:
                    has_usage = True
                    prefix += usage_chunk()
                lines.append(f'{prefix}data: [DONE]')
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                lines.append(line)
                continue
            if is_record(chunk):
                choices = chunk.get('choices')
                if not isinstance(choices, list):
                    choices = []
                    chunk['choices'] = choices
                for c in choices:
                    if not is_record(c):
                        continue
                    if isinstance(c.get('finish_reason'), str):
                        # Upstream closing twice or duplicate finish chunk: blank repeat, keep single finish
                        if saw_finish:
                            c['finish_reason'] = None
                        else:
                            saw_finish = True
                    delta = c.get('delta')
                    if is_record(delta):
                        if isinstance(delta.get('content'), str):
                            completion_chars += len(delta['content'])
                        if isinstance(delta.get('tool_calls'), list):
                            for tc in delta['tool_calls']:
                                if is_record(tc) and is_record(tc.get('function')) and isinstance(tc['function'].get('arguments'), str):
                                    completion_chars += len(tc['function']['arguments'])
                if is_record(chunk.get('usage')):
                    has_usage = True
                    pt = chunk['usage']['prompt_tokens'] if is_number(chunk['usage'].get('prompt_tokens')) and chunk['usage']['prompt_tokens'] >= 0 else calc_prompt
                    ct = chunk['usage']['completion_tokens'] if is_number(chunk['usage'].get('completion_tokens')) and chunk['usage']['completion_tokens'] >= 1 else max(1, math.ceil(completion_chars / 4))
                    chunk['usage'] = {
                        **chunk['usage'],
                        'prompt_tokens': int(pt),
                        'completion_tokens': int(ct),
                        'total_tokens': int(pt + ct),
                    }
                lines.append(f'data: {json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False)}')
            else:
                lines.append(line)
        return '\n'.join(lines) + '\n\n'

    for frame in iter_frames(chunks):
        yield process_frame(frame)

    if not saw_finish:
        yield finish_chunk()
    if not has_usage:
        has_usage = True
        yield usage_chunk()
    if not saw_done:
        saw_done = True
        yield 'data: [DONE]\n\n'


def anthropic_events_from_openai(chunks, model: str, req_prompt_tokens: int | None = None):
    """Chat Completions SSE stream translated to Anthropic SSE sequence."""
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    default_input_tokens = req_prompt_tokens if req_prompt_tokens and req_prompt_tokens > 0 else 10
    input_tokens = default_input_tokens
    output_tokens = 0
    total_output_chars = 0

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_stream',
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': int(input_tokens), 'output_tokens': 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        final_output_tokens = int(output_tokens) if output_tokens >= 1 else max(1, math.ceil(total_output_chars / 4))
        effective_stop = 'tool_use' if len(tool_indexes) > 0 else stop_reason
        return [
            *close_events(),
            sse_event(
                'message_delta',
                {'type': 'message_delta', 'delta': {'stop_reason': effective_stop, 'stop_sequence': None}, 'usage': {'output_tokens': final_output_tokens}},
            ),
            sse_event('message_stop', {'type': 'message_stop'}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop, total_output_chars
        out: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                pt = chunk['usage'].get('prompt_tokens')
                ct = chunk['usage'].get('completion_tokens')
                if is_number(pt) and pt >= 0:
                    input_tokens = int(pt)
                if is_number(ct) and ct >= 1:
                    output_tokens = int(ct)
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                total_output_chars += len(delta['content'])
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event('content_block_start', {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})
                    )
                out.append(
                    sse_event(
                        'content_block_delta',
                        {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': delta['content']}},
                    )
                )
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_idx = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                idx = tool_indexes.get(upstream_idx)
                if idx is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    idx = next_index
                    next_index += 1
                    tool_indexes[upstream_idx] = idx
                    fn = call.get('function') if is_record(call.get('function')) else {}
                    out.append(
                        sse_event(
                            'content_block_start',
                            {
                                'type': 'content_block_start',
                                'index': idx,
                                'content_block': {
                                    'type': 'tool_use',
                                    'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{idx}',
                                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                    'input': {},
                                },
                            },
                        )
                    )
                fn = call.get('function') if is_record(call.get('function')) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    total_output_chars += len(fn['arguments'])
                    out.append(
                        sse_event(
                            'content_block_delta',
                            {'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}},
                        )
                    )
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice['finish_reason'])
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else 'end_turn'):
            yield event


# ---------------------------------------------------------------- Failover & Gateway


class Upstream:
    """One gateway response wrapper with safe chunking, timeout protection, and single close."""

    def __init__(self, status: int, response, text_body: str | None = None) -> None:
        self.status = status
        self._response = response
        self._text_body = text_body
        self._closed = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """Yield byte chunks without waiting for buffer fill."""
        if self._closed or self._response is None:
            return
        reader = getattr(self._response, 'read1', None) or self._response.read
        try:
            while not self._closed:
                chunk = reader(65536)
                if not chunk:
                    break
                yield chunk
        except (socket.timeout, TimeoutError, BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.close()

    def text(self) -> str:
        if self._text_body is not None:
            return self._text_body
        if self._closed or self._response is None:
            return ''
        try:
            return self._response.read().decode('utf-8', 'replace')
        except Exception:
            return ''
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway Chat Completions route with explicit timeout."""
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps({**body, 'model': model}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT)
        return Upstream(response.status, response)
    except urllib.error.HTTPError as err:
        return Upstream(err.code, err)


def upstream_code(status: int, text: str) -> str:
    try:
        parsed = json.loads(text)
        if is_record(parsed):
            error = parsed.get('error')
            if is_record(error):
                code = error.get('code')
                if isinstance(code, str) and code != '':
                    return code
                err_type = error.get('type')
                if isinstance(err_type, str) and err_type != '':
                    return err_type
            code = parsed.get('code')
            if isinstance(code, str) and code != '':
                return code
    except Exception:
        pass
    return f'http_{status}'


def failure_trail(trail: list[Json]) -> dict[str, str]:
    if not trail or os.environ.get('BIMA_TRAIL') != '1':
        return {}
    value = json.dumps(trail[:8], separators=(',', ':'), ensure_ascii=False)
    if len(value.encode('utf-8')) > 8192:
        return {}
    return {'x-makefun-upstream-failures': value}


# ---------------------------------------------------------------- HTTP Server


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/1'
    sys_version = ''

    def log_message(self, format: str, *args: object) -> None:
        pass

    def log_error(self, format: str, *args: object) -> None:
        if args and any(isinstance(a, str) and ('Broken pipe' in a or 'Connection reset' in a) for a in args):
            return
        log(format % args)

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path in ('/v1/chat/completions', '/v1/responses'):
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, extra_headers: dict[str, str] | None = None) -> None:
        try:
            text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
            raw = text.encode('utf-8')
            self.send_response(status)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(raw)))
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.headers_sent = True
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type, None))

    def stream_headers(self, extra_headers: dict[str, str] | None = None) -> None:
        try:
            self.send_response(200)
            self.send_header('content-type', 'text/event-stream')
            self.send_header('cache-control', 'no-cache')
            self.send_header('connection', 'close')
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.headers_sent = True
            self.end_headers()
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def write_event(self, text: str) -> None:
        try:
            self.wfile.write(text.encode('utf-8'))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            raise

    def read_json(self) -> Json | None:
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        try:
            raw = self.rfile.read(size) if size > 0 else b''
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            return None
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire: str, status: int, text: str) -> None:
        suffix = '' if text == '' else f': {text[:500]}'
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', f'the gateway answered HTTP {status}{suffix}')
        else:
            self.send_error_body(wire, 502, 'upstream_error', f'the gateway answered HTTP {status}{suffix}')

    def models(self) -> None:
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        req_tokens = estimate_prompt_tokens(body)
        candidates = route_candidates(requirements_for(body))
        if not candidates:
            self.send_error_body('openai', 503, 'no_free_model', 'no free model in the brief meets this request')
            return

        route: Json = candidates[0]
        upstream: Upstream | None = None
        failure: dict[str, object] = {'status': 502, 'text': 'no candidate answered'}
        trail: list[Json] = []

        # Failover across candidates: attempt dial before sending anything to client
        for candidate in candidates:
            candidate_id = candidate['id']
            try:
                attempt = dial(candidate_id, {**body, 'model': candidate_id})
                if attempt.ok:
                    upstream = attempt
                    route = candidate
                    break
                fail_text = attempt.text()
                failure = {'status': attempt.status, 'text': fail_text}
                trail.append({'status': attempt.status, 'code': upstream_code(attempt.status, fail_text), 'model': candidate_id})
                with model_failures_lock:
                    model_failures[candidate_id] = model_failures.get(candidate_id, 0) + 1
            except Exception as err:
                failure = {'status': 502, 'text': str(err)}
                trail.append({'status': 502, 'code': 'upstream_error', 'model': candidate_id})
                with model_failures_lock:
                    model_failures[candidate_id] = model_failures.get(candidate_id, 0) + 1

        if upstream is None:
            self.refused('openai', int(failure['status']), str(failure['text']))
            return

        recovered = failure_trail(trail)

        # After first byte is sent to client, stream must complete or close without retrying
        if body.get('stream') is True:
            self.stream_headers(extra_headers=recovered)
            try:
                for frame in openai_stream_with_model(upstream.chunks(), route['id'], req_tokens):
                    self.write_event(frame)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            finally:
                upstream.close()
            return

        answer_text = upstream.text()
        try:
            parsed = json.loads(answer_text)
        except (json.JSONDecodeError, ValueError):
            parsed = {}
        self.send_json(200, openai_json_with_model(parsed if is_record(parsed) else {}, route['id'], req_tokens), extra_headers=recovered)

    def messages(self) -> None:
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        req_tokens = estimate_prompt_tokens(body)
        candidates = route_candidates(requirements_for(body))
        if not candidates:
            self.send_error_body('anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request')
            return

        route: Json = candidates[0]
        upstream: Upstream | None = None
        failure: dict[str, object] = {'status': 502, 'text': 'no candidate answered'}
        trail: list[Json] = []

        # Failover across candidates: attempt dial before sending anything to client
        for candidate in candidates:
            candidate_id = candidate['id']
            try:
                openai_body = anthropic_to_openai(body, candidate_id)
                attempt = dial(candidate_id, openai_body)
                if attempt.ok:
                    upstream = attempt
                    route = candidate
                    break
                fail_text = attempt.text()
                failure = {'status': attempt.status, 'text': fail_text}
                trail.append({'status': attempt.status, 'code': upstream_code(attempt.status, fail_text), 'model': candidate_id})
                with model_failures_lock:
                    model_failures[candidate_id] = model_failures.get(candidate_id, 0) + 1
            except Exception as err:
                failure = {'status': 502, 'text': str(err)}
                trail.append({'status': 502, 'code': 'upstream_error', 'model': candidate_id})
                with model_failures_lock:
                    model_failures[candidate_id] = model_failures.get(candidate_id, 0) + 1

        if upstream is None:
            self.refused('anthropic', int(failure['status']), str(failure['text']))
            return

        recovered = failure_trail(trail)

        # After first byte is sent to client, stream must complete or close without retrying
        if body.get('stream') is True:
            self.stream_headers(extra_headers=recovered)
            try:
                for event in anthropic_events_from_openai(upstream.chunks(), route['id'], req_tokens):
                    self.write_event(event)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            finally:
                upstream.close()
            return

        answer_text = upstream.text()
        try:
            parsed = json.loads(answer_text)
        except (json.JSONDecodeError, ValueError):
            parsed = {}
        self.send_json(200, openai_to_anthropic(parsed if is_record(parsed) else {}, route['id'], req_tokens), extra_headers=recovered)


# ---------------------------------------------------------------- Register & Poll


def register(port: int, report: Json) -> bool:
    """POST /register with port and catalog. Retries up to 3 times with backoff <= 1s, giving up cleanly."""
    if not ENV_URL:
        log('registration failed: MAKEFUN_ENV_URL is not set (giving up)')
        return False

    body = json.dumps(
        {'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report},
        separators=(',', ':'),
        ensure_ascii=False,
    ).encode('utf-8')
    url = f'{ENV_URL}/register'
    last_err = ''

    for attempt in range(1, MAX_REG_ATTEMPTS + 1):
        req = urllib.request.Request(
            url,
            data=body,
            headers={'content-type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                status = response.status
                reply_text = response.read().decode('utf-8', 'replace')
                if 200 <= status < 300:
                    try:
                        reply = json.loads(reply_text)
                        if is_record(reply) and is_number(reply.get('requests')):
                            log(f'registered: the trial holds {reply["requests"]} prompts')
                        else:
                            log('registered: ok')
                    except Exception:
                        log('registered: ok')
                    return True
                last_err = f'HTTP {status}: {reply_text[:200]}'
        except urllib.error.HTTPError as err:
            last_err = f'HTTP {err.code}: {err.read().decode("utf-8", "replace")[:200]}'
        except Exception as err:
            last_err = str(err)

        if attempt < MAX_REG_ATTEMPTS:
            time.sleep(0.5)

    log(f'registration failed after {MAX_REG_ATTEMPTS} attempts: {last_err} (giving up)')
    return False


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment done is True."""
    if not ENV_URL:
        return
    url = f'{ENV_URL}/work'
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status != 200:
                    continue
                raw = response.read().decode('utf-8', 'replace')
                work = json.loads(raw)
                if is_record(work) and work.get('done') is True:
                    sys.stderr.flush()
                    sys.stdout.flush()
                    os._exit(0)
        except Exception:
            continue


# ---------------------------------------------------------------- Start & Main


def stop(*_: object) -> None:
    """SIGTERM and SIGINT handler: exit 0 cleanly and immediately."""
    try:
        sys.stderr.flush()
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(0)


def start() -> None:
    if '--list' in sys.argv[1:]:
        # Print catalog JSON to stdout, exit 0 in < 10s without touching network
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return

    if BRIEF.get('error') is not None:
        log(str(BRIEF['error']))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    server.daemon_threads = True
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(served_models())} free models')
        registered = register(port, CATALOG)
        if ENV_URL and registered:
            threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
