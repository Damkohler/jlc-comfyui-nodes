const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
    constructor(tag) {this.tag = tag;this.style = {};this.children = [];this.events = {};this.isConnected = false;this.clientWidth = 0;}
    append(...items) {this.children.push(...items);}
    appendChild(item) {this.children.push(item);return item;}
    replaceChildren(fragment) {this.children = fragment.children;}
    setAttribute() {}
    addEventListener(name, handler) {this.events[name] = handler;}
    getClientRects() {return [];}
}

async function fixture(raw, count, connectedSlot = null) {
    let extension;const frames = [], alerts = [];
    const context = {app: {registerExtension: e => extension = e}, console,
        window: {alert: text => alerts.push(text)},
        document: {createElement: tag => new Element(tag), createTextNode: text => ({text}), createDocumentFragment: () => new Element('fragment')},
        ResizeObserver: class {observe() {} disconnect() {}}, requestAnimationFrame: f => frames.push(f)};
    vm.createContext(context);
    const source = fs.readFileSync(path.join(__dirname, '../web/jlc_text_concatenate.js'), 'utf8').replace(/^import .*;\r?\n/m, '');
    vm.runInContext(source, context);
    class Node {}
    await extension.beforeRegisterNodeDef(Node, {name: 'JLC_TextConcatenate'});
    const storage = {name: 'sections', value: raw}, countWidget = {name: 'number_of_sections', value: count};
    const node = Object.assign(new Node(), {widgets: [storage, countWidget], size: [200, 100],
        outputs: Array.from({length: 33}, (_, i) => ({name: i ? `section_${i}` : 'text', type: 'STRING', links: i === connectedSlot ? [42] : null})),
        graph: {setDirtyCanvas() {}}, computeSize() {throw Error('Must not measure a hidden editor');},
        addWidget(type, name, value, callback) {const w = {type, name, value, callback};this.widgets.push(w);return w;},
        addDOMWidget() {return {};}, addOutput(name, type) {this.outputs.push({name, type});},
        removeOutput(index) {assert.equal(this.outputs[index].links, null);this.outputs.splice(index, 1);}});
    node.onNodeCreated();while (frames.length) frames.shift()();
    return {node, storage, countWidget, alerts, update() {node.widgets.find(w => w.name === 'Update Slots').callback();while (frames.length) frames.shift()();}};
}

test('legacy arrays restore fixed slot names and discard unapplied count changes', async () => {
    const f = await fixture('["first","second"]', 10);
    assert.equal(f.countWidget.value, 2);
    assert.deepEqual(f.node.outputs.map(o => o.name), ['text', 'section_01', 'section_02']);
    assert.deepEqual(JSON.parse(f.storage.value).map(s => s.text), ['first', 'second']);
});

test('shrinking refuses to discard nonempty text or a connected output', async () => {
    const f = await fixture('["first","second"]', 2, 2);
    const outputs = [...f.node.outputs];f.countWidget.value = 1;f.update();
    assert.equal(f.countWidget.value, 2);assert.deepEqual(f.node.outputs, outputs);assert.equal(f.alerts.length, 1);
});

test('connected output beyond stored rows is preserved during workflow restore', async () => {
    const f = await fixture('["first"]', 1, 3);
    assert.equal(f.countWidget.value, 3);assert.deepEqual(f.node.outputs[3].links, [42]);
    assert.equal(JSON.parse(f.storage.value)[0].text, 'first');
});

test('invalid saved data is left byte-for-byte intact', async () => {
    const raw = '{broken';const f = await fixture(raw, 1);assert.equal(f.storage.value, raw);
});

test('slot growth retains previous socket identities and section text', async () => {
    const f = await fixture('["first"]', 1);const original = f.node.outputs[1];
    f.countWidget.value = 3;f.update();assert.equal(f.node.outputs[1], original);
    assert.deepEqual(f.node.outputs.map(o => o.name), ['text', 'section_01', 'section_02', 'section_03']);
    assert.equal(JSON.parse(f.storage.value)[0].text, 'first');
});
