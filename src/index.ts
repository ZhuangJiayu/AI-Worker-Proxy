import { Env, OpenAIChatRequest, ProviderConfig } from './types';
import { AnthropicRequest } from './anthropic-types';
import { Router } from './router';
import {
  ProxyError,
  createErrorResponse,
  withTimeout,
  isRetryableError,
  shouldRotateKey,
} from './utils/error-handler';
import { createProvider } from './providers';
import { AnthropicProvider } from './providers/anthropic';
import {
  convertAnthropicRequestToOpenAI,
  convertOpenAIResponseToAnthropic,
  AnthropicStreamAdapter,
} from './utils/anthropic-adapter';
import { generateId } from './utils/response-mapper';

const CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key',
  'Access-Control-Max-Age': '86400',
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    try {
      const url = new URL(request.url);
      const path = url.pathname;

      // Health check — no auth required
      if ((path === '/health' || path === '/') && request.method === 'GET') {
        return json({
          status: 'ok',
          service: 'ai-worker-proxy',
          timestamp: new Date().toISOString(),
        });
      }

      // Everything below requires auth
      if (!verifyAuth(request, env)) {
        throw new ProxyError('Unauthorized', 401, 'invalid_auth');
      }

      // Models list (requires auth, matching OpenAI behavior)
      if (
        request.method === 'GET' &&
        (path === '/models' ||
          path === '/v1/models' ||
          path === '/anthropic/models' ||
          path === '/anthropic/v1/models')
      ) {
        const router = new Router(env);
        return json({
          object: 'list',
          data: router.getAvailableModels(),
        });
      }

      // Anthropic-format chat completions — POST only
      if (
        request.method === 'POST' &&
        (path === '/anthropic/v1/messages' || path === '/anthropic/messages')
      ) {
        return handleAnthropicChat(request, env);
      }

      // OpenAI-format chat completions — POST only
      if (
        request.method === 'POST' &&
        (path === '/' || path === '/v1/chat/completions' || path === '/chat/completions')
      ) {
        return handleChatCompletion(request, env);
      }

      throw new ProxyError('Not found', 404);
    } catch (error) {
      console.error('[Worker] Error:', error);
      const errorResponse = createErrorResponse(error);
      const headers = new Headers(errorResponse.headers);
      for (const [key, value] of Object.entries(CORS_HEADERS)) {
        headers.set(key, value);
      }
      return new Response(errorResponse.body, { status: errorResponse.status, headers });
    }
  },
};

// =============================================================================
// OpenAI-format handler (existing)
// =============================================================================

async function handleChatCompletion(request: Request, env: Env): Promise<Response> {
  const body = await request.json();
  const chatRequest = body as OpenAIChatRequest;

  if (!chatRequest.messages || !Array.isArray(chatRequest.messages)) {
    throw new ProxyError('Invalid request: messages array is required', 400);
  }
  if (!chatRequest.model) {
    throw new ProxyError('Invalid request: model is required', 400);
  }

  console.log(`[Worker] model=${chatRequest.model} stream=${chatRequest.stream || false}`);

  const router = new Router(env);
  const response = await router.executeWithFallback(chatRequest, 'openai');

  if (!response.success) {
    throw new ProxyError(response.error || 'All providers failed', response.statusCode || 500);
  }

  if (response.stream) {
    return new Response(response.stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        ...CORS_HEADERS,
      },
    });
  }

  return json(response.response);
}

// =============================================================================
// Anthropic-format handler (dual-path)
// =============================================================================

async function handleAnthropicChat(request: Request, env: Env): Promise<Response> {
  const body: AnthropicRequest = await request.json();

  if (!body.messages || !Array.isArray(body.messages)) {
    throw new ProxyError('Invalid request: messages array is required', 400);
  }
  if (!body.model) {
    throw new ProxyError('Invalid request: model is required', 400);
  }

  console.log(`[AnthropicHandler] model=${body.model} stream=${body.stream || false}`);

  const router = new Router(env);
  const providers = router.getProvidersForModel(body.model);

  // Check if any provider supports Anthropic natively
  const hasAnthropicNative = providers.some(
    (p) => p.provider === 'anthropic' || p.provider === 'anthropic-compatible'
  );

  if (hasAnthropicNative) {
    return handleAnthropicNativePath(body, providers, env);
  }

  return handleAnthropicConversionPath(body, router);
}

/**
 * Anthropic-native path: call AnthropicProvider.nativeChat() directly.
 * No format conversion — preserves Anthropic request/response format end-to-end.
 */
async function handleAnthropicNativePath(
  body: AnthropicRequest,
  providers: ProviderConfig[],
  env: Env
): Promise<Response> {
  let lastError: unknown = null;
  let lastStatusCode: number | undefined;

  // First try all anthropic-compatible providers natively
  for (const config of providers) {
    if (config.provider !== 'anthropic' && config.provider !== 'anthropic-compatible') {
      continue;
    }

    let provider: AnthropicProvider;
    try {
      provider = createProvider(config, env) as AnthropicProvider;
    } catch (error) {
      // Bad config (e.g. missing baseUrl) — skip this provider, don't abort the loop
      console.error(
        `[AnthropicNativePath] Failed to create provider ${config.provider}/${config.model}:`,
        error
      );
      lastError = error;
      continue;
    }
    const apiKeys = resolveApiKeys(config, env);

    if (apiKeys.length === 0) {
      if (config.provider === 'anthropic-compatible') {
        // Keyless local gateway (vLLM, Ollama, ...) — mirror TokenManager's
        // pass-through and let the upstream decide on auth
        apiKeys.push('');
      } else {
        console.warn(
          `[AnthropicNativePath] No API keys found for provider: ${config.provider}/${config.model}`
        );
        lastError = new Error(`No API keys configured for ${config.provider}/${config.model}`);
        continue;
      }
    }

    for (const apiKey of apiKeys) {
      try {
        const timeoutMs = Number(env.PROVIDER_TIMEOUT_MS) || undefined;
        const result = await withTimeout(provider.nativeChat(body, apiKey), timeoutMs);
        if (!result.success) {
          lastError = result.error;
          if (result.statusCode) {
            lastStatusCode = result.statusCode;
          }
          if (!shouldRotateKey(result.statusCode, result.error)) {
            break;
          }
          continue;
        }

        if (result.stream) {
          return new Response(result.stream, {
            headers: {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              Connection: 'keep-alive',
              ...CORS_HEADERS,
            },
          });
        }

        return json(result.rawResponse);
      } catch (error: unknown) {
        lastError = error;
        // SDK errors carry the HTTP status on `status` (not `statusCode`) — check both
        const status =
          (error as { status?: number; statusCode?: number } | null)?.status ??
          (error as { statusCode?: number } | null)?.statusCode;
        if (status) {
          lastStatusCode = status;
        }
        if (!isRetryableError(error)) {
          break;
        }
      }
    }
  }

  // Fall back to conversion path for any non-Anthropic providers
  const otherProviders = providers.filter(
    (p) => p.provider !== 'anthropic' && p.provider !== 'anthropic-compatible'
  );
  if (otherProviders.length > 0) {
    const router = new Router(env);
    return handleAnthropicConversionPath(body, router, otherProviders);
  }

  const err = lastError as { message?: string } | null;
  throw new ProxyError(`All providers failed: ${err?.message || lastError}`, lastStatusCode ?? 500);
}

/**
 * Conversion path: convert Anthropic request → OpenAI → route → convert response back.
 * Used when the target provider does not support Anthropic natively.
 */
async function handleAnthropicConversionPath(
  body: AnthropicRequest,
  router: Router,
  overrideProviders?: ProviderConfig[]
): Promise<Response> {
  const openaiRequest = convertAnthropicRequestToOpenAI(body);

  const response = await router.executeWithFallback(openaiRequest, 'anthropic', overrideProviders);

  if (!response.success) {
    throw new ProxyError(response.error || 'All providers failed', response.statusCode || 500);
  }

  if (response.stream) {
    const adapter = new AnthropicStreamAdapter(body.model, `msg_${generateId()}`);
    const anthropicStream = response.stream.pipeThrough(adapter.createTransformStream());
    return new Response(anthropicStream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        ...CORS_HEADERS,
      },
    });
  }

  if (!response.response) {
    throw new ProxyError('Provider returned success but no response data', 502);
  }

  return json(convertOpenAIResponseToAnthropic(response.response, body.model));
}

// =============================================================================
// Helpers
// =============================================================================

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
  });
}

function verifyAuth(request: Request, env: Env): boolean {
  if (!env.PROXY_AUTH_TOKEN) {
    console.error('[verifyAuth] PROXY_AUTH_TOKEN not configured — rejecting all requests');
    return false;
  }

  // OpenAI-style: Authorization: Bearer <token>
  const authHeader = request.headers.get('Authorization');
  if (authHeader) {
    const token = authHeader.startsWith('Bearer ') ? authHeader.substring(7) : authHeader;
    if (constantTimeEqual(token, env.PROXY_AUTH_TOKEN)) return true;
  }

  // Anthropic-style: x-api-key: <token>
  const apiKey = request.headers.get('x-api-key');
  if (apiKey !== null && constantTimeEqual(apiKey, env.PROXY_AUTH_TOKEN)) return true;

  return false;
}

/**
 * Constant-time string comparison for the shared secret token.
 * Length inequality still short-circuits — acceptable for a fixed-length token.
 */
function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

function resolveApiKeys(config: ProviderConfig, env: Env): string[] {
  const keys: string[] = [];
  for (const keyName of config.apiKeys ?? []) {
    const value = env[keyName];
    if (value) {
      keys.push(value);
    } else {
      console.warn(`[resolveApiKeys] API key not found in env: ${keyName}`);
    }
  }
  return keys;
}
