'use strict';

const assert = require('node:assert/strict');
const { LocalDecisionModel, LocalML } = require('@rust-ml-runtime/node');

const modelRoot = process.env.ML_RUNTIME_MODEL_DIR;
assert(modelRoot, 'ML_RUNTIME_MODEL_DIR must point to the installed model root');
assert.equal(process.platform, 'linux', 'this smoke test is Linux-specific');

const direct = new LocalDecisionModel('laya', modelRoot);
const description = JSON.parse(direct.descriptionJson());
assert.equal(description.backend, 'onnx');
assert.match(description.identifier, /laya/);
assert.equal(description.artifact_sha256.length, 64);

const decisions = [
  {
    name: 'department',
    instructions: 'Which department should handle this request?',
    kind: {
      type: 'choice',
      options: [
        { label: 'billing', description: 'refunds, invoices, and duplicate charges' },
        { label: 'technical', description: 'product or account malfunction' },
        { label: 'sales', description: 'new purchases and upgrades' },
      ],
    },
  },
  {
    name: 'urgency',
    instructions: 'How urgent is this request?',
    kind: {
      type: 'score',
      levels: ['routine', 'soon', 'urgent'],
    },
  },
  {
    name: 'refund',
    instructions: 'Does the customer request a refund?',
    kind: {
      type: 'noul',
      false_description: 'not requesting a refund',
      true_description: 'requesting a refund',
    },
  },
];

const runtime = new LocalML({ modelRoot });
const input = 'The customer needs a refund for a duplicate invoice and would like help today.';

(async () => {
  const capabilities = runtime.capabilities('laya');
  assert.equal(capabilities.backend, 'onnx');
  assert.equal(capabilities.device, 'cpu');
  assert.equal(capabilities.supports_batching, true);
  assert.equal(capabilities.max_batch_size, 16);
  const nodes = decisions.map((question) => ({ question }));
  const plan = runtime.explainDecision({ model: 'laya', input, nodes });
  assert.deepEqual(plan.strategy, { type: 'batched', batch_size: 3 });

  const result = await runtime.decideAsync({ model: 'laya', input, decisions });

  assert.equal(result.backend, 'onnx');
  assert.deepEqual(result.decisions.map((decision) => decision.kind), ['choice', 'score', 'noul']);
  assert.equal(result.planning.strategy.type, 'batched');
  assert.equal(result.planning.executed_nodes, 3);
  assert.equal(typeof result.decisions[0].value.value, 'string');
  assert.equal(typeof result.decisions[1].value.value, 'number');
  assert.equal(typeof result.decisions[2].value.value, 'boolean');
  for (const decision of result.decisions) {
    assert(Number.isFinite(decision.confidence));
    assert(Number.isFinite(decision.action_probability));
    const probabilities = Object.values(decision.probabilities);
    assert(probabilities.length >= 2);
    assert(probabilities.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));
    assert(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-5);
  }
  assert.equal(result.provenance.artifact_sha256, description.artifact_sha256);

  const selfTest = LocalML.selfTest({ model: 'laya', modelRoot });
  assert.equal(selfTest.available, true, JSON.stringify(selfTest));
  assert.equal(selfTest.model.backend, 'onnx');

  console.log(JSON.stringify({
    ok: true,
    platform: `${process.platform}-${process.arch}`,
    backend: result.backend,
    model: result.model,
    capabilities,
    planning: result.planning,
    decisions: result.decisions,
    provenance: result.provenance,
  }));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
