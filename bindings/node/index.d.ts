export interface ModelDescription {
  identifier: string;
  revision?: string;
  backend: string;
  artifact_path: string;
  artifact_sha256: string;
  decision_types: string[];
}

export class LocalDecisionModel {
  constructor(model: string, modelRoot?: string);
  descriptionJson(): string;
  capabilitiesJson(): string;
  explainDecisionJson(requestJson: string): string;
  decideJson(requestJson: string): string;
  decideAsyncJson(
    requestJson: string,
    policyJson?: string,
    cancellation?: DecisionCancellation,
  ): Promise<string>;
  executeGraphJson(requestJson: string, cancellation?: DecisionCancellation): Promise<string>;
}

export class DecisionCancellation {
  constructor();
  cancel(): void;
  readonly isCancelled: boolean;
}

export interface LocalMLOptions {
  modelRoot?: string;
}

export interface NativeDiagnostics {
  platform: string;
  packageName: string | null;
  packageInstalled: boolean;
  packageResolved: string | null;
  nativeBinary: string | null;
  nativeBinaryPresent: boolean;
  load: "success" | "failure";
  status: "available" | "unavailable";
  code?: string;
  error?: Error;
  available: boolean;
}

export class NativeLoadError extends Error {
  code: string;
  diagnostics: NativeDiagnostics;
  diagnostic: NativeDiagnostics;
}

export function diagnoseNative(): NativeDiagnostics;

/** Executes a minimal native CPU inference used by release compatibility tests. */
export function runtimeSmokeTest(): string;

export interface LocalMLSelfTestResult {
  available: boolean;
  platform: string;
  model?: ModelDescription;
  checks: Array<{
    check: string;
    ok: boolean;
    message: string;
    code?: string;
  }>;
}

export interface LocalMLDecisionRequest {
  model: string;
  input: unknown;
  decisions: Array<{
    name: string;
    instructions: string;
    kind: Record<string, unknown>;
  }>;
}

export interface DecisionExecutionPolicy {
  latency_target?: { secs: number; nanos: number };
  max_parallelism?: number;
  max_batch_size?: number;
  allow_parallel?: boolean;
  allow_batching?: boolean;
  allow_selective_execution?: boolean;
}

export type DecisionDependencyCondition =
  | { type: "completed" }
  | { type: "choice_equals"; value: string }
  | { type: "noul_equals"; value: boolean }
  | { type: "score_at_least"; value: number }
  | { type: "score_at_most"; value: number };

export interface DecisionNode {
  question: LocalMLDecisionRequest["decisions"][number];
  dependencies?: Array<{
    decision: string;
    condition?: DecisionDependencyCondition;
  }>;
}

export interface LocalMLDecisionGraphRequest {
  model: string;
  input: unknown;
  nodes: DecisionNode[];
  policy?: DecisionExecutionPolicy;
  cancellation?: DecisionCancellation;
}

export interface DecisionModelCapabilities {
  backend: string;
  device: string | { other: string };
  model_architecture: string | { other: string };
  supports_batching: boolean;
  max_batch_size?: number;
  batch_size?: number;
  supports_async: boolean;
  supports_cancellation: boolean;
  supports_parallel_execution: boolean;
  recommended_parallelism?: number;
  supports_structured_decisions: boolean;
}

export type DecisionExecutionStrategy =
  | { type: "single" }
  | { type: "sequential" }
  | { type: "parallel"; concurrency: number }
  | { type: "batched"; batch_size: number }
  | { type: "selective" };

export interface DecisionExecutionPlan {
  strategy: DecisionExecutionStrategy;
  reason: string;
  stages: Array<{ nodes: string[]; batches: string[][] }>;
}

export interface LocalMLDecisionResult {
  model: { identifier: string; revision?: string };
  backend: string;
  decisions: Array<{
    name: string;
    kind: string;
    value: { type: string; value: unknown };
    probabilities: Record<string, number>;
    confidence: number;
    action_probability: number;
  }>;
  execution: {
    latency: { secs: number; nanos: number };
    input_tokens: number;
    output_tokens: number;
  };
  provenance: {
    artifact_path: string;
    artifact_sha256: string;
    runtime_version: string;
  };
  planning?: {
    capabilities: DecisionModelCapabilities;
    strategy: DecisionExecutionStrategy;
    reason: string;
    requested_nodes: number;
    executed_nodes: number;
    skipped_nodes: number;
    batches: string[][];
    max_concurrency: number;
    cancelled: boolean;
  };
}

export class LocalML {
  static create(options?: LocalMLOptions): Promise<LocalML>;
  static selfTest(options?: LocalMLOptions & { model?: string }): LocalMLSelfTestResult;
  decide(request: LocalMLDecisionRequest): LocalMLDecisionResult;
  decideAsync(request: LocalMLDecisionRequest & {
    policy?: DecisionExecutionPolicy;
    cancellation?: DecisionCancellation;
  }): Promise<LocalMLDecisionResult>;
  capabilities(model: string): DecisionModelCapabilities;
  explainDecision(request: LocalMLDecisionGraphRequest): DecisionExecutionPlan;
  executeGraph(request: LocalMLDecisionGraphRequest): Promise<LocalMLDecisionResult>;
}

// ---- Generative inference (OpenAI-compatible endpoints) --------------------------

export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export type ResponseFormat =
  | { type: "text" }
  | { type: "json_object" }
  | { type: "json_schema"; name: string; schema: unknown; strict?: boolean };

export interface GenerationOptions {
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string[];
  seed?: number;
  response_format?: ResponseFormat;
}

export interface ChatRequest {
  /** Explicit model identity. The runtime never substitutes another model. */
  model: string;
  messages: ChatMessage[];
  options?: GenerationOptions;
}

/** `null`/absent means the provider did not report it; nothing is estimated. */
export type FinishReason = "stop" | "length" | "content_filter" | { other: string };

export interface Usage {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
}

/** Execution evidence recorded by the runtime, never by the model. */
export interface ChatMetadata {
  provider: string;
  transport: string;
  execution: "local" | "remote";
  /** True when `execution` was inferred from a loopback host, not configured. */
  execution_inferred: boolean;
  /** Endpoint without credentials, query, or fragment. */
  endpoint: string;
  requested_model: string;
  reported_model: string | null;
  response_id: string | null;
  streamed: boolean;
  stream_events: number | null;
  time_to_first_token: { secs: number; nanos: number } | null;
  duration: { secs: number; nanos: number };
  runtime_version: string;
}

/**
 * Present only when a JSON response format was requested. Generated text is
 * returned verbatim; invalid output is reported, never repaired.
 */
export type StructuredOutput =
  | { status: "validated"; value: unknown }
  | { status: "invalid_json"; error: string }
  | { status: "schema_violation"; errors: string[] };

export interface ChatResponse {
  message: ChatMessage;
  finish_reason: FinishReason | null;
  usage: Usage | null;
  metadata: ChatMetadata;
  structured: StructuredOutput | null;
}

export interface ChatStreamSummary {
  finish_reason: FinishReason | null;
  usage: Usage | null;
  metadata: ChatMetadata;
  structured: StructuredOutput | null;
}

export type ChatStreamEvent =
  | { type: "started"; provider: string; model: string }
  | { type: "delta"; content: string }
  | ({ type: "completed" } & ChatStreamSummary);

export interface GenerationModelInfo {
  id: string;
  owned_by: string | null;
}

export interface GenerativeMLOptions {
  /** Base URL including the API prefix, e.g. `http://127.0.0.1:11434/v1`. */
  endpoint: string;
  /** Bearer token. Prefer `apiKeyEnv`; set at most one of the two. */
  apiKey?: string;
  /** Name of an environment variable holding the bearer token. */
  apiKeyEnv?: string;
  /** Explicit classification; loopback hosts are otherwise reported as local. */
  location?: "local" | "remote";
  /** Optional features cannot be discovered, so structured output must be declared. */
  capabilities?: {
    streaming?: boolean;
    stopSequences?: boolean;
    seed?: boolean;
    jsonObject?: boolean;
    jsonSchema?: boolean;
  };
  connectTimeoutMs?: number;
  requestTimeoutMs?: number;
  streamIdleTimeoutMs?: number;
}

export type GenerationErrorCode =
  | "invalid_request" | "model_not_found" | "provider_unavailable" | "capability_mismatch"
  | "resource_limit" | "timeout" | "cancelled" | "transport" | "protocol" | "execution"
  | "native" | string;

export class GenerationError extends Error {
  /** Stable machine-readable category matching the Rust `RuntimeError` variant. */
  code: GenerationErrorCode;
}

/**
 * Streams `started`, zero or more `delta`, then one `completed` event. Errors
 * and cancellation reject the iteration and never produce `completed`.
 * Breaking out of the loop closes the stream.
 */
export class ChatStream implements AsyncIterable<ChatStreamEvent> {
  [Symbol.asyncIterator](): AsyncGenerator<ChatStreamEvent, void, undefined>;
  /** Stops the request and releases the upstream connection. Idempotent. */
  close(): void;
  cancel(): void;
}

export class GenerativeML {
  constructor(options: GenerativeMLOptions);
  static create(options: GenerativeMLOptions): Promise<GenerativeML>;
  chat(request: ChatRequest, options?: { cancellation?: DecisionCancellation }): Promise<ChatResponse>;
  chatStream(request: ChatRequest, options?: { cancellation?: DecisionCancellation }): ChatStream;
  listModels(options?: { cancellation?: DecisionCancellation }): Promise<GenerationModelInfo[]>;
}
