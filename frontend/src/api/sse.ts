import type { StreamEvent } from './types'

export interface SseHandlers {
  onEvent?: (event: StreamEvent) => void
  onDone?: (event: StreamEvent) => void
  onError?: (error: Error) => void
}

const EVENT_TYPES = [
  'status',
  'stream',
  'human_input',
  'monitor',
  'ga_event',
  'run_complete',
  'run_error',
  'done',
  'error',
  'message',
  'ping',
]

const MAX_RECONNECT_ATTEMPTS = 50
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 10000

export function subscribeJobStream(
  url: string,
  handlers: SseHandlers,
): () => void {
  let finished = false
  let source: EventSource | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let reconnectAttempts = 0

  const cleanup = () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    source?.close()
    source = null
  }

  const finish = () => {
    if (finished) return
    finished = true
    cleanup()
  }

  const scheduleReconnect = () => {
    if (finished || reconnectTimer) return

    reconnectAttempts += 1
    if (reconnectAttempts > MAX_RECONNECT_ATTEMPTS) {
      handlers.onError?.(new Error('SSE connection lost'))
      finish()
      return
    }

    const delay = Math.min(RECONNECT_BASE_MS * reconnectAttempts, RECONNECT_MAX_MS)
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connect()
    }, delay)
  }

  const handleMessage = (eventType: string) => (event: MessageEvent) => {
    if (eventType === 'ping') return

    try {
      const data = JSON.parse(event.data) as StreamEvent
      const payload = { ...data, type: eventType }

      if (eventType === 'done') {
        handlers.onEvent?.(payload)
        handlers.onDone?.(payload)
        finish()
        return
      }

      if (eventType === 'error') {
        handlers.onEvent?.(payload)
        handlers.onError?.(new Error(String(data.content ?? data.error ?? 'Stream error')))
        finish()
        return
      }

      handlers.onEvent?.(payload)
    } catch (err) {
      if (!finished) {
        handlers.onError?.(err instanceof Error ? err : new Error(String(err)))
        finish()
      }
    }
  }

  const connect = () => {
    if (finished) return

    cleanup()
    source = new EventSource(url)

    for (const eventType of EVENT_TYPES) {
      source.addEventListener(eventType, handleMessage(eventType))
    }

    source.onopen = () => {
      reconnectAttempts = 0
    }

    source.onerror = () => {
      if (finished) return
      cleanup()
      scheduleReconnect()
    }
  }

  connect()
  return finish
}
