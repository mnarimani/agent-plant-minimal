/** Model picker helpers for chat composers (Cursor-style AUTO). */

export const AUTO_MODEL = 'auto'

/** Approximate input USD / 1M tokens — used only to pick the cheapest option. */
const INPUT_COST_HINT: Record<string, number> = {
  'gpt-4o-mini': 0.15,
  'gpt-5.4-mini': 0.75,
  'gpt-5.5': 5,
  'gpt-5.4': 5,
  'gpt-4o': 3,
  'gpt-4.1-mini': 0.4,
  'gpt-4.1': 2,
  'o4-mini': 1.1,
  'o3-mini': 1.1,
}

function costHint(model: string): number {
  const exact = INPUT_COST_HINT[model]
  if (exact !== undefined) return exact
  const lower = model.toLowerCase()
  if (lower.includes('nano')) return 0.1
  if (lower.includes('mini')) return 0.5
  if (lower.includes('small')) return 0.8
  return 50
}

/** Prefer gpt-4o-mini when present; otherwise the lowest-cost catalog entry. */
export function pickCheapestModel(models: string[]): string {
  if (models.length === 0) return 'gpt-4o-mini'
  const preferred = models.find((m) => m === 'gpt-4o-mini')
  if (preferred) return preferred
  return [...models].sort((a, b) => costHint(a) - costHint(b) || a.localeCompare(b))[0]
}

export function resolveChatModel(selection: string, models: string[]): string {
  if (selection === AUTO_MODEL || !selection) {
    return pickCheapestModel(models)
  }
  if (models.includes(selection)) return selection
  return pickCheapestModel(models)
}

export function modelLabel(selection: string): string {
  if (selection === AUTO_MODEL || !selection) return 'Auto'
  return selection.replace(/^gpt-/, '')
}
