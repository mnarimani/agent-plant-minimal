/** Parse API timestamps (UTC) and format them in the viewer's region timezone. */

const TIME_ONLY = /^(\d{2}):(\d{2})(?::(\d{2}))?$/
const HAS_TZ = /(?:[zZ]|[+-]\d{2}:?\d{2})$/

function pad(value: number): string {
  return String(value).padStart(2, '0')
}

export function viewerTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
  } catch {
    return 'UTC'
  }
}

export function parseApiDate(value: string | Date | null | undefined): Date | null {
  if (value == null || value === '') return null
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value
  }

  const raw = String(value).trim()
  if (!raw) return null

  const timeOnly = TIME_ONLY.exec(raw)
  if (timeOnly) {
    const now = new Date()
    const iso = `${now.getUTCFullYear()}-${pad(now.getUTCMonth() + 1)}-${pad(now.getUTCDate())}T${timeOnly[1]}:${timeOnly[2]}:${timeOnly[3] ?? '00'}Z`
    const date = new Date(iso)
    return Number.isNaN(date.getTime()) ? null : date
  }

  const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T')
  const withZone = HAS_TZ.test(normalized) ? normalized : `${normalized}Z`
  const date = new Date(withZone)
  return Number.isNaN(date.getTime()) ? null : date
}

export function formatDateTime(
  value: string | Date | null | undefined,
  fallback = '—',
  options?: Intl.DateTimeFormatOptions,
): string {
  const date = parseApiDate(value)
  if (!date) return value ? String(value) : fallback
  return options ? date.toLocaleString(undefined, options) : date.toLocaleString()
}

export function formatDate(
  value: string | Date | null | undefined,
  options?: Intl.DateTimeFormatOptions,
  fallback = '—',
): string {
  const date = parseApiDate(value)
  if (!date) return value ? String(value) : fallback
  return date.toLocaleDateString(
    undefined,
    options ?? {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    },
  )
}

export function formatClock(
  value: string | Date | null | undefined,
  fallback = '—',
): string {
  const date = parseApiDate(value)
  if (!date) return value ? String(value) : fallback
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function formatTimeUntil(
  value: string | Date | null | undefined,
  fallback = '',
): string {
  const date = parseApiDate(value)
  if (!date) return fallback
  const minutes = Math.max(0, Math.round((date.getTime() - Date.now()) / 60_000))
  if (minutes < 1) return 'soon'
  if (minutes < 60) return `in ${minutes} min`
  const hours = Math.floor(minutes / 60)
  const rest = minutes % 60
  if (hours < 24) return rest ? `in ${hours}h ${rest}m` : `in ${hours}h`
  const days = Math.floor(hours / 24)
  return days === 1 ? 'in 1 day' : `in ${days} days`
}

export function localizeLeadingTimestamp(text: string): string {
  return text.replace(/^\[([^\]]+)\]/, (_match, raw: string) => {
    const formatted = formatClock(raw, raw)
    return `[${formatted}]`
  })
}
