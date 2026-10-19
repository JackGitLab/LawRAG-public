// 来源链接协议限制：与后端 safe_web_source_url 保持相同允许范围。
// 只允许绝对 http:/https: 且带主机；其余一律视为不可点击文本。

function escapeMarkdownLabel(text) {
  // 规整换行/回车为空格，并转义反斜杠与方括号，防止外部来源字段
  // 通过 Markdown 特殊字符构造额外链接或破坏列表结构。
  return text
    .replace(/\\/g, '\\\\')
    .replace(/\[/g, '\\[')
    .replace(/\]/g, '\\]')
    .replace(/[\r\n]+/g, ' ')
}

export function safeSourceUrl(value) {
  if (typeof value !== 'string') return ''
  const candidate = value.trim()
  if (!candidate) return ''
  // 拒绝 C0 控制字符与 DEL
  if (/[\u0000-\u001f\u007f]/.test(candidate)) return ''
  let parsed
  try {
    parsed = new URL(candidate)
  } catch {
    return ''
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return ''
  if (!parsed.hostname) return ''
  return candidate
}

// 统一的来源显示文本：title 优先，其次 site，都为空时用“联网来源”；
// site 只在 title 存在时追加（避免重复），date 始终保留。
export function sourceDisplayText(item) {
  const title = (item && item.title) || ''
  const site = (item && item.site) || ''
  const date = (item && item.date) || ''
  const parts = []
  if (title) parts.push(title)
  // site 与 title 相同时不重复显示
  if (site && site !== title) parts.push(site)
  if (!parts.length) parts.push('联网来源')
  if (date) parts.push(date)
  return parts.join(' · ')
}

export function formatWebSourceMarkdown(item) {
  const safeUrl = safeSourceUrl(item && item.url)
  const display = sourceDisplayText(item)
  if (safeUrl) {
    // 使用原生 URL 解析后的规范化 href，避免 >、换行等字符逃逸目标边界。
    const target = new URL(safeUrl).href
    return `- [${escapeMarkdownLabel(display)}](<${target}>)`
  }
  return `- ${escapeMarkdownLabel(display)}`
}
