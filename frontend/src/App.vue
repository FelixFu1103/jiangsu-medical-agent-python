<script setup>
import { onMounted, ref } from 'vue'
const api = 'http://127.0.0.1:8000'
const question = ref(''), busy = ref(false), documents = ref([]), preview = ref(null)
const messages = ref([{ role: 'assistant', content: '您好，我是江苏医保智能咨询Agent。请告诉我您要咨询的医保事项。' }])
const form = ref({ title: '', department: '江苏省医疗保障局', source_url: '', file: null })

function formatAnswer(text = '') {
  const escaped = text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  return escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^(#{1,3})\s+(.+)$/gm, '<strong class="answer-title">$2</strong>').replace(/\n/g, '<br>')
}

async function send() {
  const value = question.value.trim(); if (!value || busy.value) return
  const history = messages.value.slice(-10); messages.value.push({ role: 'user', content: value }); question.value = ''; busy.value = true
  messages.value.push({ role: 'assistant', content: '', sources: [], structured: null, agent: null }); const answerIndex = messages.value.length - 1
  try {
    const response = await fetch(`${api}/api/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: value, history }) })
    if (!response.ok) throw new Error((await response.json()).detail || '请求失败')
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = ''
    while (true) {
      const { value: chunk, done } = await reader.read(); buffer += decoder.decode(chunk || new Uint8Array(), { stream: !done })
      const events = buffer.split('\n\n'); buffer = events.pop() || ''
      for (const raw of events) {
        const name = raw.match(/^event: (.+)$/m)?.[1], data = JSON.parse(raw.match(/^data: (.+)$/m)?.[1] || '{}')
        if (name === 'delta') messages.value[answerIndex].content += data.text
        if (name === 'done') Object.assign(messages.value[answerIndex], { sources: data.sources, structured: data.structured, agent: data.agent })
        if (name === 'error') throw new Error(data.error)
      }
      if (done) break
    }
  } catch (error) { messages.value[answerIndex].content = `抱歉，${error.message}。` } finally { busy.value = false }
}
async function loadDocuments() { documents.value = await fetch(`${api}/api/knowledge`).then(r => r.json()) }
async function inspect(id) { preview.value = await fetch(`${api}/api/knowledge/${id}`).then(r => r.json()) }
async function upload() {
  const data = new FormData(); Object.entries(form.value).forEach(([key, value]) => value && data.append(key, value))
  const response = await fetch(`${api}/api/knowledge/upload`, { method: 'POST', body: data }); const result = await response.json()
  if (!response.ok) return alert(result.detail || '上传失败')
  await loadDocuments(); await inspect(result.id); alert('草稿已生成，请预览后确认发布。')
}
async function publish() {
  if (!confirm('确认正文和片段正确并发布？发布后将参与RAG检索。')) return
  const response = await fetch(`${api}/api/knowledge/${preview.value.id}/publish`, { method: 'POST' })
  if (!response.ok) return alert((await response.json()).detail)
  await loadDocuments(); await inspect(preview.value.id); alert('发布成功')
}
onMounted(loadDocuments)
</script>

<template>
  <div class="shell">
    <header><div><b>江苏医保 Agent</b><small>Python · LangChain · LangGraph · ChromaDB</small></div><span>知识库 {{ documents.length }} 份</span></header>
    <main>
      <section class="chat card">
        <div class="messages"><article v-for="(m,i) in messages" :key="i" :class="m.role"><b>{{ m.role==='user'?'我':'医' }}</b><div>
          <section v-if="m.structured" class="answer-grid"><h3>结论</h3><p>{{ m.structured.conclusion }}</p><template v-if="m.structured.steps.length"><h3>办理建议</h3><ol><li v-for="item in m.structured.steps" :key="item">{{ item }}</li></ol></template><template v-if="m.structured.confirmations.length"><h3>需要确认</h3><ul><li v-for="item in m.structured.confirmations" :key="item">{{ item }}</li></ul></template></section>
          <p v-else-if="m.content" v-html="formatAnswer(m.content)"></p><p v-else class="typing">正在生成…</p>
          <footer v-if="m.sources?.length">官方依据：<a v-for="s in m.sources" :key="s.chunkId" :href="s.url" target="_blank">{{ s.title }}</a></footer>
          <details v-if="m.agent" class="trace"><summary>Agent 执行详情</summary><p>意图：{{ m.agent.responseType }} · 工具：{{ m.agent.selectedTool || '未调用' }}</p><p>路径：{{ m.agent.trace.join(' → ') }}</p><p v-if="m.agent.evidenceReason">证据：{{ m.agent.evidenceReason }}</p></details>
        </div></article></div>
        <form class="composer" @submit.prevent="send"><textarea v-model="question" placeholder="请输入江苏医保问题"/><button :disabled="busy">发送</button></form>
      </section>
      <aside>
        <section class="card"><h2>上传知识库</h2><input v-model="form.title" placeholder="资料标题"><input v-model="form.department" placeholder="发布部门"><input v-model="form.source_url" placeholder="政府网页 HTTPS URL"><input type="file" accept=".pdf,.docx,.txt,.md,.html" @change="form.file=$event.target.files[0]"><button @click="upload">上传为草稿</button></section>
        <section class="card"><h2>资料列表</h2><button class="doc" v-for="d in documents" :key="d.id" @click="inspect(d.id)"><span>{{ d.title }}</span><em :class="d.status">{{ d.status }} · {{ d.chunks }}片段</em></button></section>
      </aside>
    </main>
    <div class="modal" v-if="preview" @click.self="preview=null"><section class="card preview"><button class="close" @click="preview=null">×</button><h2>{{ preview.title }}</h2><p>状态：{{ preview.status }} · {{ preview.chunks.length }}个向量片段</p><h3>提取正文</h3><pre>{{ preview.body }}</pre><h3>切分预览</h3><ol><li v-for="c in preview.chunks" :key="c.id">{{ c.content }}</li></ol><button v-if="preview.status==='draft'" @click="publish">确认发布</button></section></div>
  </div>
</template>

<style scoped>
.answer-grid h3{margin:10px 0 4px;color:#123c70;font-size:15px}.answer-grid h3:first-child{margin-top:0}.answer-grid ol,.answer-grid ul{margin:4px 0;padding-left:22px}.answer-grid li{margin:5px 0;line-height:1.6}.trace{margin-top:10px;padding-top:8px;border-top:1px dashed #cad6e4;font-size:12px;color:#607086}.trace summary{cursor:pointer;color:#1761bd}.trace p{margin:5px 0;line-height:1.5}
</style>
