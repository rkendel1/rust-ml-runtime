'use strict';

const path = require('node:path');

const platformPackages = {
  'darwin-arm64': '@rust-ml-runtime/node-darwin-arm64',
  'darwin-x64': '@rust-ml-runtime/node-darwin-x64',
  'linux-arm64': '@rust-ml-runtime/node-linux-arm64-gnu',
  'linux-x64': '@rust-ml-runtime/node-linux-x64-gnu',
  'win32-x64': '@rust-ml-runtime/node-win32-x64-msvc',
};

function loadNative() {
  if (process.env.ML_RUNTIME_NODE_ADDON) return require(process.env.ML_RUNTIME_NODE_ADDON);
  const platform = `${process.platform}-${process.arch}`;
  const packageName = platformPackages[platform];
  if (!packageName) throw new Error(`@rust-ml-runtime/node does not support ${platform}`);
  try {
    return require(packageName);
  } catch (error) {
    try {
      return require(path.join(__dirname, 'native', platform, 'ml_runtime_node.node'));
    } catch {
      throw new Error(
        `Could not load ${packageName} for ${platform}. Reinstall @rust-ml-runtime/node `
          + 'with optional dependencies enabled.',
        { cause: error },
      );
    }
  }
}

const native = loadNative();

class LocalML {
  constructor(options = {}) {
    this.modelRoot = options.modelRoot;
    this.models = new Map();
  }

  static async create(options = {}) {
    return new LocalML(options);
  }

  decide({ model, input, decisions }) {
    if (!model || !Array.isArray(decisions) || decisions.length === 0) {
      throw new TypeError('decide requires model and at least one typed decision');
    }
    let loaded = this.models.get(model);
    if (!loaded) {
      loaded = new native.LocalDecisionModel(model, this.modelRoot);
      this.models.set(model, loaded);
    }
    return JSON.parse(loaded.decideJson(JSON.stringify({ state: input, decisions })));
  }
}

module.exports = { ...native, LocalML };
