import { getAuthToken } from '../api/client'
import type { ErrorTrackingSettings } from '../api/types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

let config: ErrorTrackingSettings | null = null
let handlersInstalled = false
let configPromise: Promise<ErrorTrackingSettings> | null = null

export async function loadErrorTrackingConfig(): Promise<ErrorTrackingSettings> {
  if (config) return config
  if (configPromise) return configPromise
  configPromise = (async () => {
    try {
      const response = await fetch(`${API_BASE}/errors/config`)
      if (!response.ok) {
        config = { enabled: false, frontend: false, backend: false, api: false }
        return config
      }
      config = (await response.json()) as ErrorTrackingSettings
      return config
    } catch {
      config = { enabled: false, frontend: false, backend: false, api: false }
      return config
    } finally {
      configPromise = null
    }
  })()
  return configPromise
}

export function getErrorTrackingConfig(): ErrorTrackingSettings | null {
  return config
}

export function setErrorTrackingConfig(next: ErrorTrackingSettings): void {
  config = next
}

const MAX_REPORTS_PER_WINDOW = 5
const WINDOW_MS = 10_000
const DEDUP_MS = 5_000

const recentReportTimestamps: number[] = []
const recentReportHashes = new Map<string, number>()

function isRateLimitedOrDuplicate(message: string, path?: string | null): boolean {
  const now = Date.now()
  while (recentReportTimestamps.length > 0 && now - recentReportTimestamps[0] > WINDOW_MS) {
    recentReportTimestamps.shift()
  }
  if (recentReportTimestamps.length >= MAX_REPORTS_PER_WINDOW) {
    return true
  }

  const key = `${path ?? ''}::${message}`
  const lastTime = recentReportHashes.get(key)
  if (lastTime && now - lastTime < DEDUP_MS) {
    return true
  }

  recentReportTimestamps.push(now)
  recentReportHashes.set(key, now)

  if (recentReportHashes.size > 50) {
    for (const [k, t] of recentReportHashes.entries()) {
      if (now - t > DEDUP_MS) recentReportHashes.delete(k)
    }
  }

  return false
}

export function shouldReportFrontendErrors(): boolean {
  return Boolean(config?.enabled && config.frontend)
}

export function reportFrontendError(payload: {
  message: string
  stack_trace?: string | null
  path?: string | null
  method?: string | null
  status_code?: number | null
  page_url?: string | null
  extra?: Record<string, unknown> | null
}): void {
  if (!shouldReportFrontendErrors()) return
  if (isRateLimitedOrDuplicate(payload.message, payload.path)) return

  const token = getAuthToken()
  void fetch(`${API_BASE}/errors/report`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      message: payload.message.slice(0, 4096),
      stack_trace: payload.stack_trace?.slice(0, 16384) ?? null,
      path: payload.path?.slice(0, 512) ?? null,
      method: payload.method?.slice(0, 16) ?? null,
      status_code: payload.status_code ?? null,
      page_url: payload.page_url?.slice(0, 1024) ?? window.location.href,
      extra: payload.extra ?? null,
    }),
  }).catch(() => {
    /* never break the app for reporting failures */
  })
}

export function installGlobalErrorHandlers(): void {
  if (handlersInstalled || !shouldReportFrontendErrors()) return
  handlersInstalled = true

  window.addEventListener('error', (event) => {
    if (!shouldReportFrontendErrors()) return
    const message =
      event.message ||
      (event.error instanceof Error ? event.error.message : 'Unhandled error')
    const stack =
      event.error instanceof Error
        ? event.error.stack
        : `${event.filename}:${event.lineno}:${event.colno}`
    reportFrontendError({
      message,
      stack_trace: stack,
      page_url: window.location.href,
      extra: { type: 'window.onerror' },
    })
  })

  window.addEventListener('unhandledrejection', (event) => {
    if (!shouldReportFrontendErrors()) return
    const reason = event.reason
    const message =
      reason instanceof Error
        ? reason.message
        : typeof reason === 'string'
          ? reason
          : 'Unhandled promise rejection'
    const stack = reason instanceof Error ? reason.stack : null
    reportFrontendError({
      message,
      stack_trace: stack,
      page_url: window.location.href,
      extra: { type: 'unhandledrejection' },
    })
  })
}

export async function initErrorTracking(): Promise<void> {
  await loadErrorTrackingConfig()
  installGlobalErrorHandlers()
}
