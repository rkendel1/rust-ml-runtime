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
  decideJson(requestJson: string): string;
}

export interface LocalMLOptions {
  modelRoot?: string;
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
}

export class LocalML {
  static create(options?: LocalMLOptions): Promise<LocalML>;
  decide(request: LocalMLDecisionRequest): LocalMLDecisionResult;
}
