<script setup>
import { ref } from 'vue'
import { safeSourceUrl, sourceDisplayText } from '../source-links.js'

defineProps({
  messages: { type: Array, required: true },
  streaming: { type: Boolean, default: false },
})

const copiedIndex = ref(null)
const copyErrorIndex = ref(null)

async function copyMessage(message, index) {
  try {
    await navigator.clipboard.writeText(message.content)
    copiedIndex.value = index
    copyErrorIndex.value = null
    window.setTimeout(() => {
      if (copiedIndex.value === index) copiedIndex.value = null
    }, 1500)
  } catch {
    copyErrorIndex.value = index
    window.setTimeout(() => {
      if (copyErrorIndex.value === index) copyErrorIndex.value = null
    }, 1500)
  }
}
</script>

<template>
  <div class="messages">
    <article
      v-for="(message, index) in messages"
      :key="index"
      class="message"
      :class="message.role"
    >
      <div class="content">
        {{ message.content }}<span
          v-if="streaming && index === messages.length - 1 && message.role === 'assistant'"
          class="cursor"
          aria-hidden="true"
        />
      </div>
      <ul v-if="message.role === 'assistant' && message.citations?.length" class="citations">
        <li v-for="(item, citationIndex) in message.citations" :key="citationIndex">
          《{{ item.source }}》{{ item.article }}（相关度：{{ Number(item.score).toFixed(4) }}）
        </li>
      </ul>
      <p v-else-if="message.role === 'assistant' && message.citations" class="muted">暂无引用来源</p>
      <ul v-if="message.role === 'assistant' && message.web_sources?.length" class="web-sources">
        <li v-for="(item, sourceIndex) in message.web_sources" :key="sourceIndex">
          <a
            v-if="safeSourceUrl(item.url)"
            :href="safeSourceUrl(item.url)"
            target="_blank"
            rel="noopener noreferrer"
          >
            {{ sourceDisplayText(item) }}
          </a>
          <span v-else>
            {{ sourceDisplayText(item) }}
          </span>
        </li>
      </ul>
      <div v-if="message.role === 'assistant'" class="copy-row">
        <button type="button" class="copy" @click="copyMessage(message, index)">
          {{ copiedIndex === index ? '已复制' : '复制' }}
        </button>
        <span v-if="copyErrorIndex === index" class="copy-error">复制失败</span>
      </div>
    </article>
  </div>
</template>

<style scoped>
.messages { display: flex; flex-direction: column; gap: 28px; width: min(760px, 100%); margin: 0 auto; }
.message { max-width: 80%; }
.user { align-self: flex-end; padding: 12px 16px; border-radius: var(--radius-lg); background: var(--bubble-user-bg); color: var(--bubble-user-text); }
.assistant { align-self: flex-start; color: var(--text); }
.content { white-space: pre-wrap; font-size: 16px; line-height: 1.7; }
.citations { margin: 14px 0 0; padding: 12px 0 0 22px; border-top: 1px solid var(--line); color: var(--text-secondary); font-size: 13px; line-height: 1.6; }
.web-sources { margin: 10px 0 0; padding: 10px 0 0 22px; border-top: 1px solid var(--line); color: var(--text-secondary); font-size: 13px; line-height: 1.6; }
.web-sources a { color: var(--accent); text-decoration: none; }
.web-sources a:hover { text-decoration: underline; }
.muted { margin: 10px 0 0; color: var(--text-secondary); font-size: 13px; }
.copy-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
.copy { padding: 4px 0; border: 0; background: transparent; color: var(--text-secondary); font-size: 12px; transition: color .2s var(--ease); }
.copy:hover { color: var(--accent); }
.copy-error { color: var(--danger); font-size: 12px; }
.cursor { display: inline-block; width: 2px; height: 1em; margin-left: 2px; background: var(--accent); vertical-align: -2px; animation: blink .9s steps(1) infinite; }
@keyframes blink { 50% { opacity: 0; } }
</style>
