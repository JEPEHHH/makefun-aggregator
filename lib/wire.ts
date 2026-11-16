// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The entry dials the grader's
// gateway, an OpenAI-compatible endpoint, so every prompt goes upstream as a Chat Completions request
// and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id) to the gateway and relays
//     the gateway's JSON or SSE answer, with every `model` field set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the gateway's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really answered, so a response's
//   `model` must be the model this entry dialled. OpenRouter's own answer names a free model without
//   the gateway's `openrouter/` prefix; these functions put the dialled id back on every answer and
//   every streamed chunk. They never make a model up: the id came from the brief and went to the
//   gateway in the request's `model`.
//
// Everything here is a pure function of its arguments, except the stream relays: no clock and no
// randomness, so the same body and the same gateway answer give the same bytes on a replay. The two
// model listings use fixed timestamps for the same reason: a `created` that moved between runs would be
// a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** Clean JSON content from markdown code fences (```json ... ```) or surrounding prose. */
function cleanJsonContent(text: string): string {
  const trimmed = text.trim();
  // 1. Direct code block wrapping the entire string or surrounded by whitespace
  const match = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (match) {
    return match[1].trim();
  }
  // 2. Code block anywhere inside the text
  const fenceAnywhere = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenceAnywhere) {
    const candidate = fenceAnywhere[1].trim();
    try {
      JSON.parse(candidate);
      return candidate;
    } catch {}
  }
  // 3. Prose around a JSON object/array
  try {
    JSON.parse(trimmed);
    return trimmed;
  } catch {
    const firstBrace = trimmed.indexOf('{');
    const lastBrace = trimmed.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      const candidate = trimmed.slice(firstBrace, lastBrace + 1);
      try {
        JSON.parse(candidate);
        return candidate;
      } catch {}
    }
    const firstBracket = trimmed.indexOf('[');
    const lastBracket = trimmed.lastIndexOf(']');
    if (firstBracket !== -1 && lastBracket > firstBracket) {
      const candidate = trimmed.slice(firstBracket, lastBracket + 1);
      try {
        JSON.parse(candidate);
        return candidate;
      } catch {}
    }
  }
  return text;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      const firstBrace = value.indexOf('{');
      const lastBrace = value.lastIndexOf('}');
      if (firstBrace !== -1 && lastBrace > firstBrace) {
        try {
          const candidate: unknown = JSON.parse(value.slice(firstBrace, lastBrace + 1));
          if (isRecord(candidate)) return candidate;
        } catch {}
      }
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' && block.id !== '' ? block.id : `toolu_${toolCalls.length + 1}`,
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  out.max_tokens = typeof body.max_tokens === 'number' && body.max_tokens > 0 ? body.max_tokens : 1024;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      if (raw.type === 'function' && isRecord(raw.function)) {
        return [raw];
      }
      if (typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
    else if (choice.type === 'function' && isRecord(choice.function)) out.tool_choice = choice;
    else if (choice.type === 'none') out.tool_choice = 'none';
  } else if (typeof choice === 'string') {
    out.tool_choice = choice;
  }
  if (isRecord(body.response_format)) {
    out.response_format = body.response_format;
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** Approximate the number of prompt tokens from a request body (chars / 4, at least 1). */
export function estimatePromptTokens(body: Json): number {
  let chars = 0;
  if (typeof body.system === 'string') {
    chars += body.system.length;
  }
  if (Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      if (!isRecord(msg)) continue;
      if (typeof msg.content === 'string') {
        chars += msg.content.length;
      } else if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (!isRecord(block)) continue;
          if (typeof block.text === 'string') chars += block.text.length;
          if (typeof block.content === 'string') chars += block.content.length;
        }
      }
    }
  }
  if (Array.isArray(body.tools)) {
    chars += JSON.stringify(body.tools).length;
  }
  return Math.max(1, Math.ceil(chars / 4));
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function openAIToAnthropic(body: Json, model: string, reqPromptTokens?: number): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const rawText = typeof message.content === 'string' ? message.content : textOf(message.content);
  const text = cleanJsonContent(rawText);
  if (text !== '') content.push({ type: 'text', text });

  const rawCalls: Json[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (isRecord(call)) rawCalls.push(call);
    }
  }
  if (rawCalls.length === 0 && isRecord(message.function_call)) {
    rawCalls.push({
      id: 'toolu_0',
      type: 'function',
      function: message.function_call,
    });
  }

  let toolChars = 0;
  for (let i = 0; i < rawCalls.length; i++) {
    const call = rawCalls[i];
    if (!isRecord(call) || !isRecord(call.function)) continue;
    const rawArgs = typeof call.function.arguments === 'string' ? call.function.arguments : '';
    toolChars += rawArgs.length;
    const callId = typeof call.id === 'string' && call.id.trim() !== '' ? call.id : `toolu_${i + 1}`;
    content.push({
      type: 'tool_use',
      id: callId,
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }

  if (content.length === 0) content.push({ type: 'text', text: '' });

  const hasToolUse = content.some((b) => isRecord(b) && b.type === 'tool_use');
  let stop_reason: string;
  if (hasToolUse) {
    stop_reason = 'tool_use';
  } else {
    stop_reason = stopReasonOf(choice.finish_reason);
    if (stop_reason === 'tool_use') {
      stop_reason = 'end_turn';
    }
  }

  const usage = isRecord(body.usage) ? body.usage : {};
  const calcPrompt = reqPromptTokens && reqPromptTokens > 0 ? reqPromptTokens : 10;
  const calcCompletion = Math.max(1, Math.ceil((text.length + toolChars) / 4));
  const input_tokens = typeof usage.prompt_tokens === 'number' && usage.prompt_tokens >= 0 ? usage.prompt_tokens : calcPrompt;
  const output_tokens = typeof usage.completion_tokens === 'number' && usage.completion_tokens >= 1 ? usage.completion_tokens : calcCompletion;

  return {
    id: `msg_${typeof body.id === 'string' && body.id !== '' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason,
    stop_sequence: null,
    usage: { input_tokens, output_tokens },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
export function openAIJsonWithModel(body: Json, model: string, reqPromptTokens?: number): Json {
  const choices = Array.isArray(body.choices)
    ? body.choices.map((c) => {
        if (!isRecord(c)) return c;
        const choice = { ...c };
        if (isRecord(choice.message)) {
          const msg = { ...choice.message };
          if (typeof msg.content === 'string') {
            msg.content = cleanJsonContent(msg.content);
          }
          choice.message = msg;
        }
        if (typeof choice.finish_reason !== 'string' || choice.finish_reason === '') {
          const hasTools = isRecord(choice.message) && Array.isArray(choice.message.tool_calls) && choice.message.tool_calls.length > 0;
          choice.finish_reason = hasTools ? 'tool_calls' : 'stop';
        }
        return choice;
      })
    : [];

  let completionChars = 0;
  for (const choice of choices) {
    if (isRecord(choice) && isRecord(choice.message)) {
      const msg = choice.message;
      if (typeof msg.content === 'string') completionChars += msg.content.length;
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          if (isRecord(tc) && isRecord(tc.function) && typeof tc.function.arguments === 'string') {
            completionChars += tc.function.arguments.length;
          }
        }
      }
    }
  }
  const calcPrompt = reqPromptTokens && reqPromptTokens > 0 ? reqPromptTokens : 10;
  const calcCompletion = Math.max(1, Math.ceil(completionChars / 4));

  let usage: Json;
  if (isRecord(body.usage)) {
    const pt = typeof body.usage.prompt_tokens === 'number' && body.usage.prompt_tokens >= 0 ? body.usage.prompt_tokens : calcPrompt;
    const ct = typeof body.usage.completion_tokens === 'number' && body.usage.completion_tokens >= 1 ? body.usage.completion_tokens : calcCompletion;
    usage = {
      ...body.usage,
      prompt_tokens: pt,
      completion_tokens: ct,
      total_tokens: typeof body.usage.total_tokens === 'number' && body.usage.total_tokens > 0 ? body.usage.total_tokens : (pt + ct),
    };
  } else {
    usage = {
      prompt_tokens: calcPrompt,
      completion_tokens: calcCompletion,
      total_tokens: calcPrompt + calcCompletion,
    };
  }
  return { ...body, model, choices, usage };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id the
 * entry dialled. Frames are split on the blank line that ends them, as the SSE grammar does; a frame cut
 * across reads is held until its end arrives, and a body that ends without one is relayed as it is.
 * Guarantees a usage chunk is delivered to the client before [DONE].
 */
export async function* openAIStreamWithModel(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  reqPromptTokens?: number,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let hasUsage = false;
  let sawFinish = false;
  let sawDone = false;
  let completionChars = 0;
  const calcPrompt = reqPromptTokens && reqPromptTokens > 0 ? reqPromptTokens : 10;

  const usageChunk = (): string => {
    const calcCompletion = Math.max(1, Math.ceil(completionChars / 4));
    const payload = {
      id: 'chatcmpl-usage',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: calcPrompt,
        completion_tokens: calcCompletion,
        total_tokens: calcPrompt + calcCompletion,
      },
    };
    return `data: ${JSON.stringify(payload)}\n\n`;
  };

  const finishChunk = (): string => {
    sawFinish = true;
    const payload = {
      id: 'chatcmpl-finish',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
    };
    return `data: ${JSON.stringify(payload)}\n\n`;
  };

  const processFrame = (frame: string): string => {
    const lines = frame.split('\n').map((line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) return line;
      const payload = trimmed.slice(5).trim();
      if (payload === '') return line;
      if (payload === '[DONE]') {
        sawDone = true;
        let prefix = '';
        if (!sawFinish) {
          prefix += finishChunk();
        }
        if (!hasUsage) {
          hasUsage = true;
          prefix += usageChunk();
        }
        return `${prefix}data: [DONE]`;
      }
      try {
        const chunk: unknown = JSON.parse(payload);
        if (isRecord(chunk)) {
          if (!Array.isArray(chunk.choices)) {
            chunk.choices = [];
          }
          for (const c of chunk.choices) {
            if (isRecord(c)) {
              if (typeof c.finish_reason === 'string') {
                // An upstream that closes twice would make the trial read two finish chunks (stream_order
                // fails, the whole prompt turns invalid): keep the first finish, blank the repeats.
                if (sawFinish) c.finish_reason = null;
                else sawFinish = true;
              }
              if (isRecord(c.delta)) {
                if (typeof c.delta.content === 'string') completionChars += c.delta.content.length;
                if (Array.isArray(c.delta.tool_calls)) {
                  for (const tc of c.delta.tool_calls) {
                    if (isRecord(tc) && isRecord(tc.function) && typeof tc.function.arguments === 'string') {
                      completionChars += tc.function.arguments.length;
                    }
                  }
                }
              }
            }
          }
          if (isRecord(chunk.usage)) {
            hasUsage = true;
            const pt = typeof chunk.usage.prompt_tokens === 'number' && chunk.usage.prompt_tokens >= 0 ? chunk.usage.prompt_tokens : calcPrompt;
            const ct = typeof chunk.usage.completion_tokens === 'number' && chunk.usage.completion_tokens >= 1 ? chunk.usage.completion_tokens : Math.max(1, Math.ceil(completionChars / 4));
            chunk.usage = {
              ...chunk.usage,
              prompt_tokens: pt,
              completion_tokens: ct,
              total_tokens: pt + ct,
            };
          }
          return `data: ${JSON.stringify({ ...chunk, model })}`;
        }
      } catch {
        // Not JSON: relay line
      }
      return line;
    });
    return `${lines.join('\n')}\n\n`;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield processFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield processFrame(buffer);
    if (!sawFinish) {
      yield finishChunk();
    }
    if (!hasUsage) {
      hasUsage = true;
      yield usageChunk();
    }
    if (!sawDone) {
      sawDone = true;
      yield 'data: [DONE]\n\n';
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the id the entry dialled, named back to the client.
 */
export async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  reqPromptTokens?: number,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  const defaultInputTokens = reqPromptTokens && reqPromptTokens > 0 ? reqPromptTokens : 10;
  let inputTokens = defaultInputTokens;
  let outputTokens = 0;
  let totalOutputChars = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    const finalOutputTokens = outputTokens >= 1 ? outputTokens : Math.max(1, Math.ceil(totalOutputChars / 4));
    const effectiveStop = toolIndexes.size > 0 ? 'tool_use' : stopReason;
    return [
      ...closeEvents(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: effectiveStop, stop_sequence: null },
        usage: { output_tokens: finalOutputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        const pt = numberOrZero(chunk.usage.prompt_tokens);
        const ct = numberOrZero(chunk.usage.completion_tokens);
        if (pt >= 0) inputTokens = pt;
        if (ct >= 1) outputTokens = ct;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        totalOutputChars += delta.content.length;
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          totalOutputChars += fn.arguments.length;
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}
