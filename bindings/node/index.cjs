'use strict';

const path = require('node:path');

const nativePath = process.env.ML_RUNTIME_NODE_ADDON
  || path.join(__dirname, 'native', 'ml_runtime_node.node');

module.exports = require(nativePath);
