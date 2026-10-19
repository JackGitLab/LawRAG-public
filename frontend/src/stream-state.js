export function recordStreamEvent(task, event) {
  if (event.type === 'done') {
    task.terminal = 'done'
  } else if (event.type === 'error') {
    task.terminal = 'error'
    task.error = event.detail || `请求失败（${event.status}）`
  }
}

export function finishStream(task) {
  if (task.cancelled || task.terminal === 'done') return ''
  if (task.terminal === 'error') return task.error || '请求失败，请稍后重试'
  return '回答未完成，请重试'
}
