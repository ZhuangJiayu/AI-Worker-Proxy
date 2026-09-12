import { AnthropicRequest, AnthropicResponse, AnthropicContentBlock } from '../anthropic-types';
import {
  OpenAIChatRequest,
  OpenAIChatResponse,
  OpenAIMessage,
  ContentPart,
  ToolCall,
} from '../types';
import { ProxyError } from './error-handler';

/**
 * Convert an Anthropic-format request to OpenAI-format request.
 * Used when the target provider is NOT anthropic-compatible.
 */
export function convertAnthropicRequestToOpenAI(anthropicReq: AnthropicRequest): OpenAIChatRequest {
  const messages: OpenAIMessage[] = [];

  // Extract system message
  if (anthropicReq.system) {
    messages.push({ role: 'system', content: anthropicReq.system });
  }

  // Convert messages
  for (const msg of anthropicReq.messages) {
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role, content: msg.content });
    } else {
      // Content is an array of blocks
      const contentParts: ContentPart[] = [];
      // Accumulate text, tool_use, and tool_result blocks separately, then
      // emit them in a fixed order. Text and tool_use from one source
      // assistant message must stay in ONE OpenAI assistant message (content
      // may coexist with tool_calls), otherwise an intermediate assistant
      // message ends up between the tool_calls message and its tool result
      // messages, which OpenAI rejects with
      // "messages with role 'tool' must be a response to a preceding message
      // with 'tool_calls'".
      const toolCalls: ToolCall[] = [];
      const toolResults: OpenAIMessage[] = [];
      for (const block of msg.content) {
        if (block.type === 'text' && block.text) {
          contentParts.push({ type: 'text', text: block.text });
        } else if (block.type === 'image' && block.source) {
          // Convert Anthropic image block to OpenAI image_url format
          if (block.source.type === 'url' && block.source.url) {
            // URL-source image — pass through directly
            contentParts.push({ type: 'image_url', image_url: { url: block.source.url } });
          } else if (block.source.media_type && block.source.data) {
            // Base64 image — convert to OpenAI data URI
            contentParts.push({
              type: 'image_url',
              image_url: {
                url: `data:${block.source.media_type};base64,${block.source.data}`,
              },
            });
          }
          // Malformed source (missing fields) — skip silently rather than
          // emitting a garbage `data:undefined;base64,undefined` URI
        } else if (block.type === 'tool_result') {
          // Tool results in Anthropic are content blocks in user messages;
          // in OpenAI they are separate messages with role 'tool'
          const toolContent =
            typeof block.content === 'string'
              ? block.content
              : block.content
                ? block.content
                    .filter((c) => c.type === 'text')
                    .map((c) => c.text)
                    .join(' ')
                : '';
          toolResults.push({
            role: 'tool',
            tool_call_id: block.tool_use_id || '',
            content: toolContent,
          });
        } else if (block.type === 'tool_use' && block.name) {
          // Accumulate so parallel calls share one assistant message
          toolCalls.push({
            id: block.id || '',
            type: 'function',
            function: {
              name: block.name,
              arguments: JSON.stringify(block.input ?? {}),
            },
          });
        }
      }

      const serializeContent = (): string | ContentPart[] | null =>
        contentParts.length === 0
          ? null
          : contentParts.length === 1 && contentParts[0].type === 'text'
            ? contentParts[0].text
            : contentParts;

      // One source assistant message maps to exactly one OpenAI assistant
      // message, carrying both text and tool_calls.
      if (toolCalls.length > 0 && msg.role === 'assistant') {
        messages.push({
          role: 'assistant',
          content: serializeContent(),
          tool_calls: toolCalls,
        });
      }
      // tool messages must come first so they stay adjacent to the preceding
      // assistant tool_calls message; user text content follows after.
      messages.push(...toolResults);
      if (contentParts.length > 0 && !(toolCalls.length > 0 && msg.role === 'assistant')) {
        messages.push({ role: msg.role, content: serializeContent() });
      }
    }
  }

  const openAIReq: OpenAIChatRequest = {
    model: anthropicReq.model,
    messages,
    max_tokens: anthropicReq.max_tokens ?? 4096,
    stream: anthropicReq.stream ?? false,
  };

  if (anthropicReq.temperature !== undefined) openAIReq.temperature = anthropicReq.temperature;
  if (anthropicReq.top_p !== undefined) openAIReq.top_p = anthropicReq.top_p;
  if (anthropicReq.stop_sequences) openAIReq.stop = anthropicReq.stop_sequences;

  // Convert Anthropic tools to OpenAI tools
  if (anthropicReq.tools && anthropicReq.tools.length > 0) {
    openAIReq.tools = anthropicReq.tools.map((tool) => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description || '',
        parameters: tool.input_schema || {},
      },
    }));
  }

  // Map Anthropic tool_choice to OpenAI tool_choice
  if (anthropicReq.tool_choice) {
    const tc = anthropicReq.tool_choice;
    if (tc.type === 'auto') {
      openAIReq.tool_choice = 'auto';
    } else if (tc.type === 'any') {
      // Anthropic 'any' forces tool use — the OpenAI equivalent is 'required', not 'auto'
      openAIReq.tool_choice = 'required';
    } else if (tc.type === 'tool' && tc.name) {
      openAIReq.tool_choice = {
        type: 'function',
        function: { name: tc.name },
      };
    }
  }

  return openAIReq;
}

/**
 * Map OpenAI finish_reason to Anthropic stop_reason.
 */
function mapFinishReason(
  reason: string | null | undefined
): 'end_turn' | 'max_tokens' | 'tool_use' | null {
  switch (reason) {
    case 'stop':
      return 'end_turn';
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
      return 'tool_use';
    default:
      return null;
  }
}

/**
 * Convert an OpenAI-format chat response to Anthropic-format response.
 */
export function convertOpenAIResponseToAnthropic(
  openaiResp: OpenAIChatResponse,
  model: string
): AnthropicResponse {
  const choice = openaiResp.choices?.[0];
  if (!choice) {
    throw new ProxyError('Provider returned a response with no choices', 502);
  }
  const message = choice.message;
  const content: AnthropicContentBlock[] = [];

  // Add text content
  if (message.content) {
    const textContent =
      typeof message.content === 'string'
        ? message.content
        : message.content.map((p) => (p.type === 'text' ? p.text : '')).join(' ');
    content.push({ type: 'text', text: textContent });
  }

  // Add tool calls
  if (message.tool_calls) {
    for (const tc of message.tool_calls) {
      let parsedInput: unknown;
      try {
        parsedInput = JSON.parse(tc.function.arguments);
      } catch {
        parsedInput = {};
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input: parsedInput,
      });
    }
  }

  return {
    id: openaiResp.id,
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: mapFinishReason(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: openaiResp.usage?.prompt_tokens || 0,
      output_tokens: openaiResp.usage?.completion_tokens || 0,
    },
  };
}

/**
 * Adapter that converts OpenAI SSE stream chunks to Anthropic SSE events.
 * Used when the target provider is NOT anthropic-compatible.
 */
export class AnthropicStreamAdapter {
  private encoder = new TextEncoder();
  private decoder = new TextDecoder();
  /** Monotonically increasing counter for the next content block index */
  private blockIndex = 0;
  /** Index of the currently active content block (used by stopBlock and deltas) */
  private activeBlockIndex = 0;
  private started = false;
  /** Set once a finish_reason chunk has been processed */
  private finished = false;
  private textBlockActive = false;
  private toolBlockActive = false;
  /** Maps OpenAI tool call index → Anthropic content block index */
  private toolBlockIndexMap = new Map<number, number>();

  constructor(
    private model: string,
    private messageId: string
  ) {}

  private encodeSSE(event: string, data: unknown): Uint8Array {
    return this.encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }

  private startMessage(controller: TransformStreamDefaultController): void {
    controller.enqueue(
      this.encodeSSE('message_start', {
        type: 'message_start',
        message: {
          id: this.messageId,
          type: 'message',
          role: 'assistant',
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      })
    );
  }

  private startTextBlock(controller: TransformStreamDefaultController): void {
    this.activeBlockIndex = this.blockIndex++;
    this.textBlockActive = true;
    controller.enqueue(
      this.encodeSSE('content_block_start', {
        type: 'content_block_start',
        index: this.activeBlockIndex,
        content_block: { type: 'text', text: '' },
      })
    );
  }

  private stopBlock(controller: TransformStreamDefaultController): void {
    controller.enqueue(
      this.encodeSSE('content_block_stop', {
        type: 'content_block_stop',
        index: this.activeBlockIndex,
      })
    );
    this.textBlockActive = false;
    this.toolBlockActive = false;
  }

  createTransformStream(): TransformStream<Uint8Array, Uint8Array> {
    let lineBuffer = '';

    return new TransformStream({
      transform: (chunk, controller) => {
        // Use { stream: true } for proper multi-byte UTF-8 handling across chunks
        const text = lineBuffer + this.decoder.decode(chunk, { stream: true });
        const lines = text.split('\n');

        // Last element may be a partial line — keep it in the buffer for the next chunk
        lineBuffer = lines.pop() || '';

        for (const line of lines) {
          this.processSSELine(line, controller);
        }
      },
      flush: (controller) => {
        // Process any remaining buffered data
        if (lineBuffer) {
          this.processSSELine(lineBuffer, controller);
        }
        lineBuffer = '';
        // Upstream ended without a finish_reason chunk — close gracefully so
        // Anthropic clients don't hang waiting for message_stop
        if (this.started && !this.finished) {
          if (this.textBlockActive || this.toolBlockActive) {
            this.stopBlock(controller);
          }
          controller.enqueue(
            this.encodeSSE('message_delta', {
              type: 'message_delta',
              delta: { stop_reason: 'end_turn', stop_sequence: null },
              usage: { output_tokens: 0 },
            })
          );
          controller.enqueue(this.encodeSSE('message_stop', { type: 'message_stop' }));
        }
      },
    });
  }

  /**
   * Process a single SSE line.
   */
  private processSSELine(line: string, controller: TransformStreamDefaultController): void {
    if (!line.startsWith('data: ')) return;
    const payload = line.slice(6).trim();
    if (payload === '[DONE]') return;

    let parsed;
    try {
      parsed = JSON.parse(payload);
    } catch {
      // Invalid JSON in SSE — skip this line
      return;
    }

    // Upstream-injected error (provider stream catch blocks) — surface it to
    // the client instead of letting the following finish look like a clean end
    if (parsed.error) {
      if (!this.started) {
        this.started = true;
        this.startMessage(controller);
      }
      controller.enqueue(
        this.encodeSSE('error', {
          type: 'error',
          error: {
            type: 'stream_error',
            message:
              typeof parsed.error.message === 'string'
                ? parsed.error.message
                : 'Upstream stream error',
          },
        })
      );
      return;
    }

    const choice = parsed.choices?.[0];
    if (!choice) return;

    // Some OpenAI-compatible providers omit `delta` on the final chunk
    const delta = choice.delta ?? {};
    const finish_reason = choice.finish_reason;

    // Start message on first chunk
    if (!this.started) {
      this.started = true;
      this.startMessage(controller);
    }

    // Text content delta
    if (delta.content) {
      // Text arrived while a tool block is active (e.g. Gemini interleaves
      // text after function calls) — close it and start a fresh text block
      if (this.toolBlockActive) {
        this.stopBlock(controller);
      }
      if (!this.textBlockActive) {
        this.startTextBlock(controller);
      }
      controller.enqueue(
        this.encodeSSE('content_block_delta', {
          type: 'content_block_delta',
          index: this.activeBlockIndex,
          delta: { type: 'text_delta', text: delta.content },
        })
      );
    }

    // Tool call delta
    if (delta.tool_calls) {
      for (const tc of delta.tool_calls) {
        if (tc.id) {
          // Close current block (text or tool) before starting a new one
          if (this.textBlockActive || this.toolBlockActive) {
            this.stopBlock(controller);
          }

          this.activeBlockIndex = this.blockIndex++;
          this.toolBlockActive = true;
          if (tc.index !== undefined) {
            this.toolBlockIndexMap.set(tc.index, this.activeBlockIndex);
          }

          controller.enqueue(
            this.encodeSSE('content_block_start', {
              type: 'content_block_start',
              index: this.activeBlockIndex,
              content_block: {
                type: 'tool_use',
                id: tc.id,
                name: tc.function?.name || '',
                input: {},
              },
            })
          );
        }

        if (tc.function?.arguments !== undefined) {
          const blockIdx =
            tc.index !== undefined
              ? (this.toolBlockIndexMap.get(tc.index) ?? this.activeBlockIndex)
              : this.activeBlockIndex;
          controller.enqueue(
            this.encodeSSE('content_block_delta', {
              type: 'content_block_delta',
              index: blockIdx,
              delta: {
                type: 'input_json_delta',
                partial_json: tc.function.arguments,
              },
            })
          );
        }
      }
    }

    // Finish reason
    if (finish_reason) {
      this.finished = true;
      if (this.textBlockActive || this.toolBlockActive) {
        this.stopBlock(controller);
      }

      controller.enqueue(
        this.encodeSSE('message_delta', {
          type: 'message_delta',
          delta: {
            stop_reason: mapFinishReason(finish_reason),
            stop_sequence: null,
          },
          usage: { output_tokens: 0 },
        })
      );

      controller.enqueue(this.encodeSSE('message_stop', { type: 'message_stop' }));
    }
  }
}
