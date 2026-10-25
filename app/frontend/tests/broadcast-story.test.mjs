import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {broadcastShape, outputValue, parseShape} from '../src/lessons/broadcastModel.js';
const fixtures=JSON.parse(fs.readFileSync(new URL('./broadcast_expected.json',import.meta.url)));
test('new article shape checker agrees with all stored NumPy fixtures',()=>{
 for(const {inputs,expect} of fixtures){const r=broadcastShape(inputs.shapeA,inputs.shapeB);assert.equal(!r.error,expect.valid);if(expect.valid){const expected=expect.result_shape.slice(-Math.max(inputs.shapeA.length,inputs.shapeB.length));assert.deepEqual(r.shape,expected);}}
});
test('all 24 animated output cells agree with NumPy',()=>{const expected=fixtures[0].expect.output;for(let i=0;i<2;i++)for(let j=0;j<4;j++)for(let k=0;k<3;k++)assert.equal(outputValue(i,j,k),expected[i][j][k]);});
test('empty dimensions and invalid shape input are handled',()=>{assert.deepEqual(broadcastShape([0,3],[1,3]).shape,[0,3]);assert.ok(broadcastShape([0,3],[2,3]).error);assert.equal(parseShape('2, NaN'),null);assert.equal(parseShape('-1,3'),null);assert.deepEqual(parseShape(' 2, 1,3 '),[2,1,3]);});
