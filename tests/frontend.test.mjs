/**
 * Pruebas de regresión sin dependencias externas.
 * No consultan Databricks, no ejecutan Jobs y no requieren secretos.
 * Ejecutar: node --test tests/frontend.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import vm from 'node:vm';

const html = readFileSync(resolve('azure/frontend/index.html'), 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)];
const script = scripts.map((m) => m[1]).join('\n');
const keys = [
  'Ciudad', 'Localidad', 'Sexo', 'RangoEdad', 'Escolaridad', 'EstadoCivil',
  'Etnia', 'PaisNacimiento', 'PertenenciaGrupal', 'RegistradoPor', 'Mes', 'Dia',
];

function getModelConfig() {
  const sampleMatch = script.match(/const SAMPLE\s*=\s*Object\.freeze\((\{[\s\S]*?\})\);/);
  const fieldsMatch = script.match(/const FIELDS\s*=\s*(\[[\s\S]*?\]);/);
  assert.ok(sampleMatch, 'No se encontró SAMPLE');
  assert.ok(fieldsMatch, 'No se encontró FIELDS');
  return vm.runInNewContext(`({ sample: ${sampleMatch[1]}, fields: ${fieldsMatch[1]} })`, {}, { timeout: 1000 });
}

test('HTML y módulos principales presentes', () => {
  assert.match(html, /<!doctype html>/i);
  for (const id of ['screen-predict', 'screen-train', 'predict-form', 'train-form', 'lookup-form']) {
    assert.ok(html.includes(`id="${id}"`), `Falta el elemento ${id}`);
  }
});

test('JavaScript incrustado con sintaxis válida', () => {
  assert.equal(scripts.length, 1, 'Se esperaba un único bloque JavaScript incrustado');
  const result = spawnSync(process.execPath, ['--check'], { input: script, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr || 'El JavaScript tiene un error de sintaxis');
});

test('Se conservan las tres rutas de Databricks a través de Azure', () => {
  assert.match(script, /api\(['"]\/api\/predecir['"]\s*,\s*\{\s*method:\s*['"]POST['"]/);
  assert.match(script, /api\(['"]\/api\/entrenar['"]\s*,\s*\{\s*method:\s*['"]POST['"]/);
  assert.match(script, /api\(['"]\/api\/entrenar\/['"]\s*\+\s*encodeURIComponent\(/);
});

test('Predicción conserva exactamente las 12 variables originales', () => {
  const { fields, sample } = getModelConfig();
  assert.deepEqual(Array.from(fields, ([field]) => String(field)), keys);
  assert.deepEqual(Object.keys(sample), keys);
  for (const key of keys) {
    assert.equal(typeof sample[key], 'string', `${key} debe enviarse como texto`);
    assert.ok(sample[key].trim(), `${key} no puede estar vacío en el ejemplo`);
  }
});

test('Entrenamiento preserva los 3 parámetros y sus tipos JSON', () => {
  const match = script.match(/function trainPayload\(\)\{([\s\S]*?)\}\s*function predictPayload/);
  assert.ok(match, 'No se encontró trainPayload');
  const values = { max_iteraciones: '2', tamano_lote: '1000', reiniciar_modelos: 'false' };
  const el = (id) => ({ value: values[id] });
  const fn = new Function('el', `return function(){${match[1]}}`)(el);
  assert.deepEqual(fn(), { max_iteraciones: 2, tamano_lote: 1000, reiniciar_modelos: false });
  values.reiniciar_modelos = 'true';
  assert.equal(fn().reiniciar_modelos, true);
});

test('Predicción construye el JSON de las 12 variables', () => {
  const { fields, sample } = getModelConfig();
  const match = script.match(/function predictPayload\(\)\{([\s\S]*?)\}\s*function updateRequestPreview/);
  assert.ok(match, 'No se encontró predictPayload');
  const el = (id) => ({ value: sample[id.replace('input-', '')] });
  const fn = new Function('FIELDS', 'el', `return function(){${match[1]}}`)(fields, el);
  assert.deepEqual(fn(), JSON.parse(JSON.stringify(sample)));
});

test('Ambos módulos presentan JSON enviado, respuesta, errores y botones Copiar', () => {
  for (const type of ['train', 'predict']) {
    for (const id of [`${type}-request`, `${type}-json`, `${type}-error`]) {
      assert.ok(html.includes(`id="${id}"`), `Falta ${id}`);
    }
    for (const attr of ['data-copy-request', 'data-copy', 'data-copy-error']) {
      assert.ok(html.includes(`${attr}="${type}"`), `Falta el botón ${attr} del módulo ${type}`);
    }
  }
});
