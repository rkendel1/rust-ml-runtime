'use strict';

const { LocalML } = require('@rust-ml-runtime/node');

const state = {
  facts: {
    eligibility: 'terminated 2026-01-01 → 2026-06-30',
  },
  policy: {
    covered: true,
    prior_authorization_required: false,
  },
};

const decisions = [{
  name: 'eligibility',
  instructions: 'Is this member eligible?',
  kind: {
    type: 'noul',
    false_description: 'not eligible',
    true_description: 'eligible',
  },
}];

async function main() {
  const runtime = await LocalML.create({ modelRoot: process.env.ML_RUNTIME_MODEL_DIR });
  const inference = runtime.decide({ model: 'laya', input: state, decisions });
  const decision = inference.decisions?.[0];
  if (decision?.kind !== 'noul' || typeof decision.value?.value !== 'boolean') {
    throw new Error(`unexpected healthcare decision: ${JSON.stringify(inference)}`);
  }
  if (Object.keys(decision.probabilities ?? {}).length !== 2 || inference.backend !== 'coreml') {
    throw new Error(`missing probabilities or provenance: ${JSON.stringify(inference)}`);
  }
  console.log(JSON.stringify({
    application: { facts: state.facts, policy: state.policy },
    decision_contract: decisions[0],
    model_inference: inference,
    authority: 'Application policy and authorization remain outside the model.',
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
