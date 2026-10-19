<script setup>
import { computed, nextTick, onMounted, reactive, ref } from 'vue'
import MessageList from './components/MessageList.vue'
import SessionList from './components/SessionList.vue'
import { formatWebSourceMarkdown } from './source-links.js'
import { finishStream, recordStreamEvent } from './stream-state.js'

const question = ref('')
const webSearchEnabled = ref(false)
const messages = ref([])
const sessions = ref([])
const sessionId = ref(null)
const conversationKey = ref(`draft-${crypto.randomUUID()}`)
const error = ref('')
const sidebarOpen = ref(false)
const messageArea = ref(null)
const messageCache = new Map()
const streams = reactive(new Map())
const currentStream = computed(() => streams.get(conversationKey.value) ?? null)

async function readError(response) {
  try {
    const data = await response.json()
    return typeof data.detail === 'string' ? data.detail : '请求失败，请稍后重试'
  } catch {
    return '请求失败，请稍后重试'
  }
}

function isNearBottom() {
  const area = messageArea.value
  return !area || area.scrollHeight - area.scrollTop - area.clientHeight < 40
}

async function scrollToBottom(shouldScroll = true) {
  if (!shouldScroll) return
  await nextTick()
  const area = messageArea.value
  if (area) area.scrollTop = area.scrollHeight
}

async function refreshSessions() {
  try {
    const response = await fetch('/api/sessions')
    if (!response.ok) throw new Error(await readError(response))
    sessions.value = await response.json()
  } catch (requestError) {
    error.value = requestError.message || '会话列表加载失败'
  }
}

function newSession() {
  question.value = ''
  messages.value = []
  sessionId.value = null
  conversationKey.value = `draft-${crypto.randomUUID()}`
  error.value = ''
  sidebarOpen.value = false
}

async function selectSession(id) {
  error.value = ''
  if (messageCache.has(id)) {
    messages.value = messageCache.get(id)
    sessionId.value = id
    conversationKey.value = id
    sidebarOpen.value = false
    scrollToBottom()
    return
  }
  try {
    const response = await fetch(`/api/sessions/${id}`)
    if (!response.ok) throw new Error(await readError(response))
    const data = await response.json()
    messages.value = data.messages
    messageCache.set(id, messages.value)
    sessionId.value = id
    conversationKey.value = id
    sidebarOpen.value = false
    scrollToBottom()
  } catch (requestError) {
    error.value = requestError.message || '会话加载失败'
  }
}

async function removeSession(id) {
  if (streams.has(id)) {
    error.value = '该会话正在生成回答，请先停止后再删除'
    return
  }
  if (!window.confirm('确定删除这个会话吗？')) return
  error.value = ''
  try {
    const response = await fetch(`/api/sessions/${id}`, { method: 'DELETE' })
    if (!response.ok) throw new Error(await readError(response))
    messageCache.delete(id)
    if (sessionId.value === id) newSession()
    await refreshSessions()
  } catch (requestError) {
    error.value = requestError.message || '会话删除失败'
  }
}

function handleEvent(event, task) {
  if (event.type === 'session') {
    streams.delete(task.key)
    task.key = event.session_id
    streams.set(task.key, task)
    messageCache.set(task.key, task.messages)
    refreshSessions()
    if (messages.value === task.messages) {
      sessionId.value = task.key
      conversationKey.value = task.key
    }
  } else if (event.type === 'token') {
    task.waiting = false
    if (task.assistantIndex === null) {
      task.messages.push({ role: 'assistant', content: '' })
      task.assistantIndex = task.messages.length - 1
    }
    const isVisible = messages.value === task.messages
    const shouldScroll = isVisible && isNearBottom()
    task.messages[task.assistantIndex].content += event.text
    if (isVisible) scrollToBottom(shouldScroll)
  } else if (event.type === 'done') {
    recordStreamEvent(task, event)
    if (task.assistantIndex !== null) {
      task.messages[task.assistantIndex].citations = event.citations
      task.messages[task.assistantIndex].web_sources = event.web_sources || []
    }
  } else if (event.type === 'error') {
    recordStreamEvent(task, event)
  }
}

function processBlocks(buffer, task) {
  const blocks = buffer.split('\n\n')
  const remainder = blocks.pop() ?? ''
  for (const block of blocks) {
    const line = block.split('\n').find((item) => item.startsWith('data: '))
    if (line) handleEvent(JSON.parse(line.slice(6)), task)
  }
  return remainder
}

async function ask(event) {
  if (event?.isComposing || event?.keyCode === 229) return
  const value = question.value.trim()
  if (!value || currentStream.value) return

  error.value = ''
  question.value = ''
  messages.value.push({ role: 'user', content: value })
  const targetMessages = messages.value
  const requestSessionId = sessionId.value
  const task = reactive({
    key: conversationKey.value,
    controller: new AbortController(),
    waiting: true,
    messages: targetMessages,
    assistantIndex: null,
    terminal: null,
    error: '',
    cancelled: false,
  })
  streams.set(task.key, task)
  if (requestSessionId) messageCache.set(requestSessionId, targetMessages)
  scrollToBottom()

  try {
    const body = { question: value, use_web_search: webSearchEnabled.value }
    if (requestSessionId) body.session_id = requestSessionId
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: task.controller.signal,
    })
    if (!response.ok) throw new Error(await readError(response))

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value: chunk } = await reader.read()
      if (done) break
      buffer += decoder.decode(chunk, { stream: true })
      buffer = processBlocks(buffer, task)
    }
    buffer += decoder.decode()
    if (buffer.trim()) processBlocks(`${buffer}\n\n`, task)
    const streamError = finishStream(task)
    if (streamError && messages.value === task.messages) error.value = streamError
    await refreshSessions()
  } catch (requestError) {
    if (requestError.name !== 'AbortError') {
      if (messages.value === task.messages) {
        error.value = requestError.message || '请求失败，请稍后重试'
      }
    }
  } finally {
    streams.delete(task.key)
  }
}

function stopGeneration() {
  const task = currentStream.value
  if (!task) return
  task.cancelled = true
  task.controller.abort()
}

function exportConversation() {
  const text = messages.value.map((message) => {
    if (message.role === 'user') return `## 问\n\n${message.content}`
    const citations = message.citations?.map(
      (item) => `- 《${item.source}》${item.article}（相关度 ${Number(item.score).toFixed(4)}）`,
    ).join('\n')
    const webSources = message.web_sources?.map(
      (item) => formatWebSourceMarkdown(item),
    ).join('\n')
    return `${message.content}${citations ? `\n\n${citations}` : ''}${webSources ? `\n\n${webSources}` : ''}`
  }).join('\n\n')
  const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  const now = new Date()
  const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}`
  link.href = url
  link.download = `lawrag-${stamp}.md`
  link.click()
  URL.revokeObjectURL(url)
}

onMounted(refreshSessions)
</script>

<template>
  <div class="shell">
    <aside class="sidebar" :class="{ open: sidebarOpen }">
      <button class="new-session" type="button" @click="newSession">＋ 新会话</button>
      <div class="session-scroll">
        <SessionList
          :sessions="sessions"
          :current-id="sessionId"
          @select="selectSession"
          @remove="removeSession"
        />
      </div>
      <div class="session-count">{{ sessions.length }} 个会话</div>
    </aside>

    <main class="main" :class="{ 'empty-chat': !messages.length }">
      <header class="topbar">
        <button class="menu" type="button" aria-label="打开会话列表" @click="sidebarOpen = !sidebarOpen">☰</button>
        <h1>LawRAG 法律问答演示</h1>
        <button class="secondary" type="button" :disabled="!messages.length" @click="exportConversation">导出</button>
      </header>
      <div class="disclaimer">本项目仅供技术学习与演示，回答由 AI 生成，不构成法律意见，不得用于实际法律决策</div>

      <section ref="messageArea" class="message-area" aria-live="polite">
        <div v-if="!messages.length" class="empty">
          <h2>有什么法律问题？</h2>
          <p>直接输入问题即可开始新会话。</p>
        </div>
        <MessageList
          :messages="messages"
          :streaming="Boolean(currentStream && !currentStream.waiting)"
        />
        <div
          v-if="currentStream?.waiting"
          class="loading"
          role="status"
        >正在检索并生成回答…</div>
      </section>

      <footer class="composer-wrap">
        <form class="composer" @submit.prevent="ask">
          <textarea
            v-model="question"
            :disabled="Boolean(currentStream)"
            rows="2"
            required
            aria-label="法律问题"
            placeholder="输入法律问题…"
            @keydown.enter.exact.prevent="ask"
          />
          <div class="composer-actions">
            <div class="composer-options">
              <span class="hint">Enter 发送，Shift + Enter 换行</span>
              <button
                class="web-search-toggle"
                :class="{ active: webSearchEnabled }"
                type="button"
                :aria-pressed="webSearchEnabled"
                :disabled="Boolean(currentStream)"
                @click="webSearchEnabled = !webSearchEnabled"
              >
                <span aria-hidden="true">⌕</span>
                联网搜索
              </button>
            </div>
            <button v-if="currentStream" class="danger" type="button" @click="stopGeneration">停止</button>
            <button v-else class="primary" type="submit" :disabled="!question.trim()">发送</button>
          </div>
        </form>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
      </footer>
    </main>
  </div>
</template>

<style scoped>
* { box-sizing: border-box; }
.shell { display: flex; height: 100dvh; overflow: hidden; background: linear-gradient(145deg, var(--content-start), var(--content-end)); color: var(--text); }
.sidebar { position: relative; z-index: 2; flex: 0 0 272px; width: 272px; display: flex; flex-direction: column; background: linear-gradient(180deg, var(--sidebar-start), var(--sidebar-end)); border-right: 1px solid var(--line); box-shadow: var(--shadow-divider); transition: transform .28s var(--ease); }
.new-session { margin: 16px 12px 10px; padding: 11px 14px; border: 0; border-radius: var(--radius-sm); background: var(--accent); color: var(--bubble-user-text); text-align: left; }
.session-scroll { flex: 1; min-height: 0; overflow-y: auto; }
.session-count { padding: 12px 20px 16px; border-top: 1px solid var(--line); color: var(--text-secondary); font-size: 12px; }
.main { flex: 1; min-width: 0; display: flex; flex-direction: column; background: linear-gradient(145deg, var(--content-start), var(--content-end)); }
.topbar { flex: 0 0 auto; display: flex; align-items: center; gap: 12px; padding: 14px 24px; background: var(--surface); border-bottom: 1px solid var(--line); }
h1 { flex: 1; margin: 0; font-size: 18px; }
.menu { display: none; }
.secondary, .menu { padding: 7px 12px; border: 1px solid var(--line); border-radius: var(--radius-sm); background: var(--surface); color: var(--text); }
.disclaimer { flex: 0 0 auto; padding: 8px 24px; background: var(--surface); color: var(--text-secondary); font-size: 12px; line-height: 1.45; }
.message-area { flex: 1; min-height: 0; overflow-y: auto; padding: 24px; }
.empty { display: grid; min-height: 100%; place-content: center; text-align: center; }
.empty h2 { margin: 0; font-size: 28px; }
.empty p { color: var(--text-secondary); }
.loading { width: min(760px, 100%); margin: 18px auto 0; color: var(--text-secondary); font-size: 14px; }
.composer-wrap { flex: 0 0 auto; padding: 12px 24px 16px; border-top: 1px solid var(--line); background: var(--surface); }
.composer { width: min(760px, 100%); margin: 0 auto; padding: 10px 12px; border: 1px solid var(--line); border-radius: var(--radius-md); background: var(--surface); box-shadow: var(--shadow-soft); }
.empty-chat .message-area { flex: 0 0 auto; padding-top: clamp(170px, 30vh, 300px); padding-bottom: 22px; overflow: visible; }
.empty-chat .empty { min-height: 0; }
.empty-chat .empty h2 { font-size: clamp(28px, 3vw, 38px); letter-spacing: -.03em; }
.empty-chat .composer-wrap { padding-top: 0; border-top: 0; background: transparent; }
.empty-chat .composer { box-shadow: 0 10px 36px var(--line); }
textarea { width: 100%; max-height: 160px; padding: 4px; resize: none; border: 0; outline: 0; background: transparent; color: var(--text); font: inherit; line-height: 1.5; }
textarea::placeholder { color: var(--text-secondary); }
.composer-actions { display: flex; align-items: center; justify-content: space-between; margin-top: 6px; }
.composer-options { display: flex; align-items: center; gap: 10px; min-width: 0; }
.hint { color: var(--text-secondary); font-size: 12px; }
.web-search-toggle { display: inline-flex; align-items: center; gap: 5px; padding: 5px 9px; border: 1px solid var(--line); border-radius: 999px; background: transparent; color: var(--text-secondary); font-size: 12px; }
.web-search-toggle:hover:not(:disabled) { background: var(--surface-hover); color: var(--text); }
.web-search-toggle.active { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 12%, transparent); color: var(--accent); }
.primary, .danger { padding: 7px 14px; border: 0; border-radius: var(--radius-sm); color: var(--bubble-user-text); }
.primary { background: var(--accent); }
.danger { background: var(--danger); }
.error { width: min(760px, 100%); margin: 8px auto 0; color: var(--danger); font-size: 13px; }
button { transition: opacity .2s var(--ease), background-color .2s var(--ease); }
button:disabled { cursor: not-allowed; opacity: .45; }
@media (max-width: 720px) {
  .sidebar { position: fixed; inset: 0 auto 0 0; z-index: 20; transform: translateX(-100%); }
  .sidebar.open { transform: none; }
  .menu { display: inline-block; }
  .topbar, .message-area { padding-right: 16px; padding-left: 16px; }
  .composer-wrap { padding-right: 12px; padding-left: 12px; }
  .empty-chat .message-area { padding-top: clamp(100px, 23vh, 190px); }
  .hint { font-size: 11px; }
  .composer-options { gap: 6px; }
  .web-search-toggle { padding: 5px 8px; }
}
</style>
