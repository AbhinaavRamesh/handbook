<script setup lang="ts">
import { computed, ref } from 'vue'
import { onContentUpdated, useRoute } from 'vitepress'

const route = useRoute()
const total = ref(0)
const done = ref(0)

const pct = computed(() => (total.value ? Math.round((done.value / total.value) * 100) : 0))

const storageKey = () => `handbook-progress:${route.path}`

let checkboxes: HTMLInputElement[] = []
let keys: string[] = []

// Keyed by item text + occurrence so progress survives item reordering;
// editing an item's text resets that one item, which is acceptable.
function keysFor(boxes: HTMLInputElement[]): string[] {
  const counts = new Map<string, number>()
  return boxes.map((box) => {
    const text = (box.closest('li')?.textContent ?? '').trim().slice(0, 120)
    const n = counts.get(text) ?? 0
    counts.set(text, n + 1)
    return `${text}::${n}`
  })
}

function loadChecked(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(storageKey()) ?? '[]'))
  } catch {
    return new Set()
  }
}

function saveChecked(checked: Set<string>) {
  if (checked.size) localStorage.setItem(storageKey(), JSON.stringify([...checked]))
  else localStorage.removeItem(storageKey())
}

function sync() {
  checkboxes = Array.from(
    document.querySelectorAll<HTMLInputElement>('.vp-doc input.task-list-item-checkbox')
  )
  total.value = checkboxes.length
  if (!checkboxes.length) {
    done.value = 0
    return
  }

  keys = keysFor(checkboxes)
  const checked = loadChecked()
  checkboxes.forEach((box, i) => {
    box.checked = checked.has(keys[i])
    box.onchange = () => {
      const current = loadChecked()
      if (box.checked) current.add(keys[i])
      else current.delete(keys[i])
      saveChecked(current)
      done.value = checkboxes.filter((b) => b.checked).length
    }
  })
  done.value = checkboxes.filter((b) => b.checked).length
}

function reset() {
  localStorage.removeItem(storageKey())
  checkboxes.forEach((box) => (box.checked = false))
  done.value = 0
}

onContentUpdated(sync)
</script>

<template>
  <div v-if="total > 0" class="study-progress" role="status" aria-live="polite">
    <div class="study-progress-header">
      <span>
        <strong>{{ done }} / {{ total }}</strong> completed
        <span v-if="done === total" class="study-progress-done">— done! 🎉</span>
      </span>
      <button
        v-if="done > 0"
        class="study-progress-reset"
        type="button"
        @click="reset"
      >
        Reset
      </button>
    </div>
    <div class="study-progress-track">
      <div class="study-progress-fill" :style="{ width: pct + '%' }"></div>
    </div>
  </div>
</template>

<style scoped>
.study-progress {
  position: sticky;
  top: calc(var(--vp-nav-height) + 12px);
  z-index: 10;
  margin-bottom: 24px;
  padding: 12px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background-color: var(--vp-c-bg-soft);
  backdrop-filter: blur(8px);
}

.study-progress-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
  color: var(--vp-c-text-1);
}

.study-progress-done {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.study-progress-reset {
  padding: 2px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  font-size: 12px;
  color: var(--vp-c-text-2);
  transition: color 0.2s, border-color 0.2s;
}

.study-progress-reset:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}

.study-progress-track {
  height: 6px;
  border-radius: 3px;
  background-color: var(--vp-c-default-soft);
  overflow: hidden;
}

.study-progress-fill {
  height: 100%;
  border-radius: 3px;
  background-color: var(--vp-c-brand-1);
  transition: width 0.3s ease;
}
</style>
