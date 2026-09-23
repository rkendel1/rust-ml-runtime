# Node and TypeScript

The preferred interactive structured-decision path is async:

```js
const local = await LocalML.create();
const capabilities = local.capabilities("laya");
const plan = local.explainDecision({ model: "laya", input, nodes });
const result = await local.executeGraph({ model: "laya", input, nodes });
```

`decide()` and `decideJson()` remain synchronous compatibility APIs.
`decideAsync()` and `executeGraph()` keep synchronous native inference off the
Node event loop. See the [planner contract](runtime/execution-planning.md) for
policies, graph conditions, cancellation, and diagnostics.

Install the public package as a normal application dependency:

```sh
npm install @rust-ml-runtime/node
```

The package selects an optional native package for the current OS and CPU. It
does not run Cargo during installation and does not contain every platform's
binary. Install the model once with the separately shipped CLI before loading
it from Node:

Production npm installs must retain optional dependencies:

```sh
npm ci --include=optional
```

Consumers should not install `@rust-ml-runtime/node-linux-x64-gnu` (or another
platform package) directly. The runtime resolves the matching package and
reports package, native artifact, and load failures through `diagnoseNative()`:

```js
const { diagnoseNative, LocalML } = require("@rust-ml-runtime/node");
console.log(diagnoseNative());
if (!LocalML.selfTest().available) throw new Error("local inference is unavailable");
```

```sh
ml-runtime model install laya
ml-runtime model doctor laya
```

```ts
import { LocalML } from "@rust-ml-runtime/node";

const ml = await LocalML.create();
const result = ml.decide({
  model: "laya",
  input: "The customer asks for a refund.",
  decisions: [{
    name: "refund",
    instructions: "Does the customer request a refund?",
    kind: { type: "noul", false_description: null, true_description: null },
  }],
});
```

`result.decisions[0]` contains the typed `value`, probabilities, and confidence.
The enclosing result contains the model identity, Core ML backend, inference
latency, runtime version, artifact hash, and artifact path. See
`examples/consumer` for the deliberately small external application fixture.

An abbreviated result looks like:

```json
{
  "backend": "coreml",
  "decisions": [{
    "name": "refund",
    "value": { "type": "noul", "value": true },
    "probabilities": { "false": 0.08, "true": 0.92 }
  }],
  "execution": { "latency": { "secs": 0, "nanos": 240000000 } },
  "provenance": {
    "artifact_sha256": "<verified model hash>",
    "runtime_version": "0.1.0"
  }
}
```

The Node-API addon loads the native runtime in-process. It does not spawn the
CLI and needs no Python, Rust, Cargo, pip, or source checkout at runtime.
