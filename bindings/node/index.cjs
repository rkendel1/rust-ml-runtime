'use strict';

const fs = require('node:fs');
const path = require('node:path');

const platformPackages = {
  'darwin-arm64': '@rust-ml-runtime/node-darwin-arm64',
  'darwin-x64': '@rust-ml-runtime/node-darwin-x64',
  'linux-arm64': '@rust-ml-runtime/node-linux-arm64-gnu',
  'linux-x64': '@rust-ml-runtime/node-linux-x64-gnu',
  'win32-x64': '@rust-ml-runtime/node-win32-x64-msvc',
};

class NativeLoadError extends Error {
  constructor(diagnostics) {
    super(
      `Could not load ${diagnostics.packageName ?? 'native binding'} for ${diagnostics.platform}`
        + (diagnostics.error ? `: ${diagnostics.error.message}` : ''),
      {
        cause: diagnostics.error,
      },
    );
    this.name = 'NativeLoadError';
    this.code = diagnostics.code;
    this.diagnostics = diagnostics;
    this.diagnostic = diagnostics;
  }
}

function diagnosticsFor(platform, packageName) {
  return {
    platform,
    packageName: packageName ?? null,
    packageInstalled: false,
    packageResolved: null,
    nativeBinary: null,
    nativeBinaryPresent: false,
    load: 'failure',
    status: 'unavailable',
  };
}

function logDiagnostics(diagnostics) {
  console.error(`rust-ml-runtime: platform ${diagnostics.platform}`);
  console.error(`rust-ml-runtime: native binding ${diagnostics.packageName ?? 'unsupported'}`);
  console.error(`rust-ml-runtime: package installed ${diagnostics.packageInstalled}`);
  console.error(`rust-ml-runtime: package resolved ${diagnostics.packageResolved ?? 'unresolved'}`);
  console.error(`rust-ml-runtime: native binary ${diagnostics.nativeBinary ?? 'missing'}`);
  console.error(`rust-ml-runtime: load ${diagnostics.load}`);
  if (diagnostics.error) console.error(`rust-ml-runtime: error ${diagnostics.error.message}`);
}

function resolveNative() {
  const platform = `${process.platform}-${process.arch}`;
  const packageName = platformPackages[platform];
  const diagnostics = diagnosticsFor(platform, packageName);
  if (!packageName) {
    diagnostics.code = 'UNSUPPORTED_PLATFORM';
    diagnostics.error = new Error(`@rust-ml-runtime/node does not support ${platform}`);
    return diagnostics;
  }

  let packageManifest;
  try {
    const manifestPath = require.resolve(`${packageName}/package.json`);
    diagnostics.packageInstalled = true;
    diagnostics.packageResolved = path.dirname(manifestPath);
    packageManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  } catch (error) {
    diagnostics.code = 'PACKAGE_MISSING';
    diagnostics.error = error;
    return diagnostics;
  }

  diagnostics.nativeBinary = path.resolve(diagnostics.packageResolved, packageManifest.main ?? '');
  diagnostics.nativeBinaryPresent = fs.existsSync(diagnostics.nativeBinary);
  if (!diagnostics.nativeBinaryPresent) {
    diagnostics.code = 'NATIVE_BINARY_MISSING';
    diagnostics.error = new Error(`native binary is missing at ${diagnostics.nativeBinary}`);
    return diagnostics;
  }

  try {
    diagnostics.native = require(diagnostics.nativeBinary);
    diagnostics.load = 'success';
    diagnostics.status = 'available';
    diagnostics.code = 'OK';
  } catch (error) {
    diagnostics.code = 'NATIVE_LOAD_FAILED';
    diagnostics.error = error;
  }
  return diagnostics;
}

function loadNative() {
  if (process.env.ML_RUNTIME_NODE_ADDON) {
    const diagnostics = diagnosticsFor(`${process.platform}-${process.arch}`, 'ML_RUNTIME_NODE_ADDON');
    diagnostics.packageInstalled = true;
    diagnostics.packageResolved = process.env.ML_RUNTIME_NODE_ADDON;
    diagnostics.nativeBinary = process.env.ML_RUNTIME_NODE_ADDON;
    try {
      diagnostics.native = require(process.env.ML_RUNTIME_NODE_ADDON);
      diagnostics.nativeBinaryPresent = true;
      diagnostics.load = 'success';
      diagnostics.status = 'available';
      diagnostics.code = 'OK';
    } catch (error) {
      diagnostics.code = 'NATIVE_LOAD_FAILED';
      diagnostics.error = error;
    }
    logDiagnostics(diagnostics);
    if (diagnostics.error) throw new NativeLoadError(diagnostics);
    return diagnostics.native;
  }

  const diagnostics = resolveNative();
  logDiagnostics(diagnostics);
  if (diagnostics.error) throw new NativeLoadError(diagnostics);
  return diagnostics.native;
}

const native = loadNative();

function diagnoseNative() {
  if (process.env.ML_RUNTIME_NODE_ADDON) {
    try {
      require(process.env.ML_RUNTIME_NODE_ADDON);
      return {
        ...diagnosticsFor(`${process.platform}-${process.arch}`, 'ML_RUNTIME_NODE_ADDON'),
        packageInstalled: true,
        packageResolved: process.env.ML_RUNTIME_NODE_ADDON,
        nativeBinary: process.env.ML_RUNTIME_NODE_ADDON,
        nativeBinaryPresent: true,
        load: 'success',
        status: 'available',
        code: 'OK',
        available: true,
      };
    } catch (error) {
      return {
        ...diagnosticsFor(`${process.platform}-${process.arch}`, 'ML_RUNTIME_NODE_ADDON'),
        packageInstalled: true,
        packageResolved: process.env.ML_RUNTIME_NODE_ADDON,
        nativeBinary: process.env.ML_RUNTIME_NODE_ADDON,
        nativeBinaryPresent: fs.existsSync(process.env.ML_RUNTIME_NODE_ADDON),
        code: 'NATIVE_LOAD_FAILED',
        error,
        available: false,
      };
    }
  }
  const diagnostics = resolveNative();
  return { ...diagnostics, available: diagnostics.load === 'success', native: undefined };
}

class LocalML {
  constructor(options = {}) {
    this.modelRoot = options.modelRoot;
    this.models = new Map();
  }

  static async create(options = {}) {
    return new LocalML(options);
  }

  static selfTest(options = {}) {
    const nativeDiagnostics = diagnoseNative();
    const checks = [{
      check: 'native binding',
      ok: nativeDiagnostics.available,
      message: nativeDiagnostics.available
        ? 'native binding loaded'
        : nativeDiagnostics.error?.message ?? 'native binding unavailable',
      code: nativeDiagnostics.code,
    }];
    if (options.model) {
      try {
        const model = new LocalDecisionModel(options.model, options.modelRoot);
        JSON.parse(model.descriptionJson());
        checks.push({ check: 'local inference', ok: true, message: 'model loaded' });
      } catch (error) {
        checks.push({
          check: 'local inference',
          ok: false,
          message: error.message,
          code: error.code,
        });
      }
    }
    return {
      available: checks.every((check) => check.ok),
      platform: nativeDiagnostics.platform,
      checks,
    };
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

module.exports = { ...native, LocalML, NativeLoadError, diagnoseNative };
