import assert from 'node:assert/strict'
import test from 'node:test'

import { finishStream, recordStreamEvent } from '../src/stream-state.js'

test('normal stream completion accepts exactly one done terminal event', () => {
  const task = { terminal: null, error: '', cancelled: false }

  recordStreamEvent(task, { type: 'done' })

  assert.equal(task.terminal, 'done')
  assert.equal(finishStream(task), '')
})

test('error event reports the server failure', () => {
  const task = { terminal: null, error: '', cancelled: false }

  recordStreamEvent(task, { type: 'error', status: 502, detail: '外部服务调用失败，请稍后重试' })

  assert.equal(task.terminal, 'error')
  assert.equal(finishStream(task), '外部服务调用失败，请稍后重试')
})

test('stream without done or error is incomplete', () => {
  const task = { terminal: null, error: '', cancelled: false }

  assert.equal(finishStream(task), '回答未完成，请重试')
})

test('user cancellation does not report a service failure', () => {
  const task = { terminal: null, error: '', cancelled: true }

  assert.equal(finishStream(task), '')
})
