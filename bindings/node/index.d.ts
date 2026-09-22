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
