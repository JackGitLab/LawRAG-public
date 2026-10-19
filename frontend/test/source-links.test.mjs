import assert from 'node:assert/strict'
import test from 'node:test'

import {
  formatWebSourceMarkdown,
  safeSourceUrl,
  sourceDisplayText,
} from '../src/source-links.js'

test('safeSourceUrl accepts absolute http/https with host', () => {
  assert.equal(safeSourceUrl('https://example.com/path?x=1#part'), 'https://example.com/path?x=1#part')
  assert.equal(safeSourceUrl('http://example.com'), 'http://example.com')
  assert.equal(safeSourceUrl('  https://example.com/  '), 'https://example.com/')
  assert.equal(safeSourceUrl('HTTPS://Example.com/A'), 'HTTPS://Example.com/A')
})

test('safeSourceUrl rejects unsafe protocols and shapes', () => {
  const unsafe = [
    'javascript:alert(1)',
    'JaVaScRiPt:alert(1)',
    'data:text/html,x',
    'file:///etc/passwd',
    'ftp://example.com/file',
    '//example.com/path',
    '/relative/path',
    'example.com/path',
    'https://',
    '',
    '   ',
    'https://example.com\nx',
    'https://exa\tmple.com',
    'https://example.com\rx',
    'https://example.com\u0000x',
    'https://example.com\u007f',
    // P12-R1：authority 存在但 hostname 为空，或畸形 IPv6 必须稳定拒绝
    'https://:80',
    'https://user@:443',
    'https://[broken',
    'http://:',
  ]
  for (const value of unsafe) {
    assert.equal(safeSourceUrl(value), '', `expected rejected: ${JSON.stringify(value)}`)
  }
  assert.equal(safeSourceUrl(undefined), '')
  assert.equal(safeSourceUrl(null), '')
  assert.equal(safeSourceUrl(42), '')
  assert.equal(safeSourceUrl(['https://example.com']), '')
  assert.equal(safeSourceUrl({ url: 'https://example.com' }), '')
})

test('formatWebSourceMarkdown links only safe urls', () => {
  assert.equal(
    formatWebSourceMarkdown({ title: '标题', url: 'https://example.com/a' }),
    '- [标题](<https://example.com/a>)',
  )
  assert.equal(
    formatWebSourceMarkdown({ title: '标题', url: 'javascript:alert(1)' }),
    '- 标题',
  )
  assert.equal(
    formatWebSourceMarkdown({ title: '标题', url: '' }),
    '- 标题',
  )
  assert.equal(
    formatWebSourceMarkdown({ title: '标题', url: 'data:text/html,x' }),
    '- 标题',
  )
  // 非法 URL 不得出现在 Markdown 链接目标中
  assert.ok(!formatWebSourceMarkdown({ title: '标题', url: 'javascript:alert(1)' }).includes(']('))
  assert.ok(!formatWebSourceMarkdown({ title: '标题', url: '' }).includes(']('))
  assert.ok(!formatWebSourceMarkdown({ title: '标题', url: 'file:///etc/passwd' }).includes(']('))
})

test('sourceDisplayText keeps title/site/date and never duplicates site', () => {
  assert.equal(
    sourceDisplayText({ title: '标题', site: '站点', date: '2026-09-21' }),
    '标题 · 站点 · 2026-09-21',
  )
  assert.equal(sourceDisplayText({ title: '', site: '站点', date: '2026-09-21' }), '站点 · 2026-09-21')
  assert.equal(sourceDisplayText({ title: '', site: '', date: '2026-09-21' }), '联网来源 · 2026-09-21')
  assert.equal(sourceDisplayText({ title: '', site: '', date: '' }), '联网来源')
  // site 不重复：title 与 site 相同时也只出现一次
  assert.equal(sourceDisplayText({ title: '站点', site: '站点' }), '站点')
  assert.equal(sourceDisplayText(null), '联网来源')
})

test('formatWebSourceMarkdown keeps metadata and blocks markdown injection', () => {
  // 非法 URL 仍保留 title/site/date，不产生链接目标
  const plain = formatWebSourceMarkdown({
    title: '标题',
    url: 'javascript:alert(1)',
    site: '站点',
    date: '2026-09-21',
  })
  assert.ok(plain.includes('标题'))
  assert.ok(plain.includes('站点'))
  assert.ok(plain.includes('2026-09-21'))
  assert.ok(!plain.includes(']('))
  assert.ok(!plain.includes('javascript:'))

  // title 为空时 site/date 不丢失；URL 为空不生成空链接
  assert.equal(
    formatWebSourceMarkdown({ title: '', url: '', site: '站点', date: '2026-09-21' }),
    '- 站点 · 2026-09-21',
  )
  assert.equal(formatWebSourceMarkdown({ title: '标题', url: '' }), '- 标题')

  // 外部字段含 Markdown 特殊字符不能构造额外链接或破坏列表
  const injected = formatWebSourceMarkdown({
    title: '标题](javascript:alert(1))',
    url: 'https://example.com/a',
    site: '带[方括号]站点\\反斜杠',
    date: '第一行\n第二行',
  })
  assert.ok(injected.includes('标题\\]')) // 标题中的 ] 被转义，无法提前闭合标签
  assert.ok(injected.includes('\\[方括号')) // 方括号被转义
  assert.ok(injected.includes('方括号\\]'))
  assert.ok(!injected.includes('\n')) // 换行规整为空格，不打断列表
  assert.ok(injected.startsWith('- ['))
  assert.ok(injected.includes('](<https://example.com/a>)')) // 仍生成一个安全链接
  // 标签文本内的 javascript: 不构成第二个链接目标
  assert.equal(injected.split('](<').length - 1, 1)
  assert.ok(!injected.includes('](<javascript:'))
})
