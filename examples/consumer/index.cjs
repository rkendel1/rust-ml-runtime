'use strict';

const { LocalML } = require('@rust-ml-runtime/node');

async function main() {
  const runtime = await LocalML.create({ modelRoot: process.env.ML_RUNTIME_MODEL_DIR });
  const result = runtime.decide({
    model: 'laya',
    input: 'The customer asks for a refund.',
    decisions: [{
      name: 'refund',
      instructions: 'Does the customer request a refund?',
      kind: { type: 'noul', false_description: null, true_description: null },
    }],
  });
  if (result.backend !== 'coreml' || result.decisions?.[0]?.probabilities?.true === undefined) {
    throw new Error(`unexpected local decision result: ${JSON.stringify(result)}`);
  }
  console.log(JSON.stringify(result, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
