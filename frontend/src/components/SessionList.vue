<script setup>
defineProps({
  sessions: { type: Array, required: true },
  currentId: { type: String, default: null },
  disabled: { type: Boolean, default: false },
})

defineEmits(['select', 'remove'])

function relativeTime(value) {
  const date = new Date(value)
  const minutes = Math.max(0, Math.floor((Date.now() - date.getTime()) / 60000))
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  if (minutes < 1440) return `${Math.floor(minutes / 60)} 小时前`
  return value.slice(0, 10)
}
</script>

<template>
  <ul class="sessions">
    <li
      v-for="session in sessions"
      :key="session.id"
      class="session"
      :class="{ current: session.id === currentId }"
    >
      <button class="session-main" type="button" @click="$emit('select', session.id)">
        <span class="title">{{ session.title }}</span>
        <span class="time">{{ relativeTime(session.updated_at) }}</span>
      </button>
      <button
        class="remove"
        type="button"
        :disabled="disabled"
        aria-label="删除会话"
        @click="$emit('remove', session.id)"
      >
        删除
      </button>
    </li>
  </ul>
</template>

<style scoped>
.sessions { margin: 0; padding: 0 8px; list-style: none; }
.session { position: relative; display: flex; align-items: center; border-radius: var(--radius-sm); transition: background-color .2s var(--ease); }
.session:hover, .session.current { background: var(--surface-hover); }
.session-main { min-width: 0; flex: 1; padding: 10px 42px 10px 12px; border: 0; background: transparent; color: var(--text); text-align: left; }
.title, .time { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.title { font-size: 14px; }
.time { margin-top: 4px; color: var(--text-secondary); font-size: 11px; }
.remove { position: absolute; right: 8px; padding: 5px; border: 0; background: transparent; color: var(--danger); font-size: 12px; opacity: 0; transition: opacity .2s var(--ease); }
.session:hover .remove, .remove:focus-visible { opacity: 1; }
button:disabled { cursor: not-allowed; opacity: .5; }
</style>
