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
