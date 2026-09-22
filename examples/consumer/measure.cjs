'use strict';

const { writeFileSync } = require('node:fs');
const { performance } = require('node:perf_hooks');
const { LocalDecisionModel } = require('@rust-ml-runtime/node');

const request = {
  state: 'The customer asks for a refund of a duplicate payment.',
  decisions: [{
    name: 'refund',
    instructions: 'Does the customer request a refund?',
    kind: { type: 'noul', false_description: null, true_description: null },
  }],
};

function percentile(sorted, fraction) {
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)];
}

async function main() {
  const loadStarted = performance.now();
  const model = new LocalDecisionModel('laya', process.env.ML_RUNTIME_MODEL_DIR);
  const preparedLoadMs = performance.now() - loadStarted;
  const encodedRequest = JSON.stringify(request);
  const firstStarted = performance.now();
  const first = JSON.parse(model.decideJson(encodedRequest));
  const firstMs = performance.now() - firstStarted;
  if (first.backend !== 'coreml') throw new Error('measurement did not execute Core ML');
  const warm = [];
  for (let index = 0; index < 20; index += 1) {
    const started = performance.now();
    model.decideJson(encodedRequest);
    warm.push(performance.now() - started);
  }
  warm.sort((left, right) => left - right);
  const measurements = {
    runtime_version: first.provenance.runtime_version,
    model_revision: first.model.revision,
    backend: first.backend,
    model_install_ms: Number(process.env.ML_RUNTIME_INSTALL_MS ?? 0),
    prepared_model_load_ms: preparedLoadMs,
    first_inference_ms: firstMs,
    warm_p50_ms: percentile(warm, 0.5),
    warm_p95_ms: percentile(warm, 0.95),
  };
  const encoded = `${JSON.stringify(measurements, null, 2)}\n`;
  if (process.env.ML_RUNTIME_MEASUREMENTS_FILE) {
    writeFileSync(process.env.ML_RUNTIME_MEASUREMENTS_FILE, encoded);
  }
  process.stdout.write(encoded);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
