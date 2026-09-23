import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { Link, useLocation, useSearchParams } from 'react-router-dom'
import {
  ArrowRight,
  Check,
  Clock,
  Loader2,
  PanelRightClose,
  PanelRightOpen,
  Sparkles,
} from 'lucide-react'
import { plantModelApi, triggerBlobDownload } from '../api/endpoints'
import type {
  PlantModelChatMessage,
  PlantModelConversationSummary,
  PlantModelResult,
  PlantModelSessionState,
} from '../api/types'
import { useAuth } from '../context/AuthContext'
import { AUTO_MODEL, resolveChatModel } from '../lib/modelPicker'
import { btnBase, btnCompact, btnPrimary } from '../lib/classes'
import { CodePreview } from './CodePreview'
import { ComposerModelPicker } from './ComposerModelPicker'
import { MarkdownContent } from './MarkdownContent'
import { StatusMessage } from './StatusMessage'

const SYSTEM_ARCHETYPES = [
  {
    id: 'power',
    category: 'Electromechanical',
    title: 'DC Motor Position Control',
    desc: 'Armature-controlled DC motor with current, speed and shaft angle states.',
    prompt: 'Design a DC motor position control system with current, angular velocity, and position states. Control the motor angular position by applying armature voltage, including electrical and mechanical dynamics.',
    badge: '3rd Order',
  },
  {
    id: 'ballbeam',
    category: 'Robotics & Mechanics',
    title: 'Ball and Beam',
    desc: 'Underactuated ball position control via beam tilt angle.',
    prompt: 'Design a ball and beam system where the ball position and velocity are states and the control input is the beam tilt angle. Stabilize the ball at a desired position on the beam.',
    badge: 'Underactuated',
  },
  {
    id: 'robotics',
    category: 'Robotics & Mechanics',
    title: 'Inverted Pendulum',
    desc: 'Rotary inverted pendulum balancing about the upright equilibrium.',
    prompt: 'Design a rotary inverted pendulum (not cart-pole) with pendulum angle and angular velocity as states. Stabilize the pendulum at the upright unstable equilibrium using torque at the pivot.',
    badge: 'Nonlinear',
  },
  {
    id: 'auv',
    category: 'Marine & AUV',
    title: 'AUV Pitch and Depth',
    desc: 'Heave-pitch coupled AUV dynamics with stern-plane control.',
    prompt: `I want to model the pitch and depth dynamics of an underwater vehicle (AUV) with a stern-plane actuator. The vehicle has a rigid-body pitch moment of inertia Iy = 469 kg*m^2, an added-mass pitch moment of -458 kg*m^2, a pitch damping coefficient Mq = 9826.2 kg*m^2/s, a pitch restoring coefficient Mtheta = 13719.6 kg*m^2/s^2, and a stern-plane control moment coefficient Mfs = -1575.9. Depth rate is roughly 20 times the pitch angle.

Use three states: q (pitch rate, rad/s), theta (pitch angle, rad), and z (depth, positive down, in metres). The only control input is fs (stern-plane deflection, rad).

The pitch dynamics follow q_dot = -(Mq*q + Mtheta*theta + Mfs*fs) / (Iy - Mq_added), with theta_dot = q and z_dot = mu1*theta. Regulate pitch and depth back to zero using the stern plane.

I'd also like the equivalent single transfer function from stern-plane deflection to depth, of the form -mu1*Mfs / (s*(I_e*s^2 + Mq*s + Mtheta)), plus the intermediate pitch transfer.`,
    badge: '3rd Order',
  },
]

interface PlantModelChatProps {
  model: string
  models: string[]
  onModelChange: (model: string) => void
  disabled?: boolean
  onUseModel: (result: PlantModelResult, conversationId?: number | null) => void
  continueLabel?: string
  continueIcon?: ReactNode
  isModalOpen?: boolean
}

function userInitials(user: { display_name: string | null; email: string } | null): string {
  if (!user) return 'U'
  const source = user.display_name?.trim() || user.email
  const parts = source.split(/\s+/).filter(Boolean)
  if (parts.length >= 2) {
    return `${parts[0][0] ?? ''}${parts[1][0] ?? ''}`.toUpperCase()
  }
  return source.slice(0, 2).toUpperCase()
}

function autoresize(el: HTMLTextAreaElement) {
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`
}

// --- Plant metadata formatting (sidebar) -----------------------------------
// Canonical order matches PlantCompiler.REQUIRED_METADATA_KEYS; any extra
// keys a future backend adds are still shown, appended after the known ones.
const METADATA_KEY_ORDER = [
  'states',
  'state_meanings',
  'inputs',
  'outputs',
  'state_equations',
  'parameters',
  'system_type',
  'assumptions',
]

const METADATA_LABELS: Record<string, string> = {
  states: 'States',
  state_meanings: 'State meanings',
  inputs: 'Inputs',
  outputs: 'Outputs',
  state_equations: 'State equations',
  parameters: 'Parameters',
  system_type: 'System type',
  assumptions: 'Assumptions',
}

function humanizeMetadataKey(key: string): string {
  return (
    METADATA_LABELS[key] ??
    key
      .split('_')
      .filter(Boolean)
      .map((word) => word[0].toUpperCase() + word.slice(1))
      .join(' ')
  )
}

function isEmptyMetadataValue(value: unknown): boolean {
  if (value === null || value === undefined) return true
  if (Array.isArray(value)) return value.length === 0
  if (typeof value === 'object') return Object.keys(value as Record<string, unknown>).length === 0
  if (typeof value === 'string') return value.trim().length === 0
  return false
}

function orderedMetadataEntries(metadata: Record<string, unknown>): [string, unknown][] {
  const seen = new Set<string>()
  const ordered: [string, unknown][] = []
  for (const key of METADATA_KEY_ORDER) {
    if (key in metadata) {
      ordered.push([key, metadata[key]])
      seen.add(key)
    }
  }
  for (const [key, value] of Object.entries(metadata)) {
    if (!seen.has(key)) ordered.push([key, value])
  }
  return ordered.filter(([, value]) => !isEmptyMetadataValue(value))
}

function formatScalar(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(6)))
  }
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  return String(value)
}

/** Render a single metadata value, adapting to its shape: short lists of
 * identifiers read fine inline, but longer or more sentence-like entries
 * (meanings, equations, assumptions) get their own line so they can wrap
 * without cramming the panel or forcing a horizontal scroll. */
function MetadataValue({ value }: { value: unknown }) {
  if (Array.isArray(value)) {
    const items = value.map((v) => (v !== null && typeof v === 'object' ? JSON.stringify(v) : formatScalar(v)))
    const stacked = items.length > 4 || items.some((s) => s.length > 22)
    if (stacked) {
      return (
        <ul className="m-0 flex list-none flex-col gap-1 p-0">
          {items.map((item, i) => (
            <li
              key={i}
              className="rounded-md bg-surface-elevated px-2 py-1 font-mono text-[11.5px] leading-snug text-foreground break-words"
            >
              {item}
            </li>
          ))}
        </ul>
      )
    }
    return <span className="break-words font-mono text-[12.5px] text-foreground">{items.join(', ')}</span>
  }

  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
    return (
      <ul className="m-0 flex list-none flex-col gap-1 p-0">
        {entries.map(([key, entryValue]) => (
          <li
            key={key}
            className="flex items-baseline justify-between gap-2 font-mono text-[11.5px] leading-snug"
          >
            <span className="shrink-0 text-muted">{key}</span>
            <span className="break-words text-right text-foreground">{formatScalar(entryValue)}</span>
          </li>
        ))}
      </ul>
    )
  }

  return <span className="break-words font-mono text-[12.5px] text-foreground">{formatScalar(value)}</span>
}

export function PlantModelChat({
  model: _model = AUTO_MODEL,
  models,
  onModelChange,
  disabled = false,
  onUseModel,
  continueLabel: _continueLabel = 'Configure & launch →',
  continueIcon: _continueIcon,
  isModalOpen = false,
}: PlantModelChatProps) {
  const { user } = useAuth()
  const initials = userInitials(user)
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()
  const newChatRequest = (location.state as { newChat?: number } | null)?.newChat

  const [messages, setMessages] = useState<PlantModelChatMessage[]>([])
  const [sessionState, setSessionState] = useState<PlantModelSessionState | null>(null)
  const [finalResult, setFinalResult] = useState<PlantModelResult | null>(null)
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [toastOpen, setToastOpen] = useState(false)
  const [selection, setSelection] = useState(AUTO_MODEL)
  const [recentConversations, setRecentConversations] = useState<PlantModelConversationSummary[]>([])
  const [plantPanelOpen, setPlantPanelOpen] = useState(true)
  const deepLinkHandled = useRef<string | null>(null)

  useEffect(() => {
    plantModelApi
      .listConversations()
      .then((data) => setRecentConversations(data.slice(0, 4)))
      .catch(() => {})
  }, [conversationId])

  const listRef = useRef<HTMLDivElement>(null)
  const landingInputRef = useRef<HTMLTextAreaElement>(null)
  const threadInputRef = useRef<HTMLTextAreaElement>(null)

  const inChat = messages.length > 0 || finalResult !== null
  const chatDisabled = disabled || loading
  const draft = finalResult ?? sessionState?.latest_draft ?? null
  const resolvedModel = resolveChatModel(selection, models)
  const progressPct = finalResult
    ? 100
    : sessionState?.latest_draft
      ? Math.min(90, 35 + (sessionState.draft_count || 1) * 20)
      : messages.length > 0
        ? Math.min(30, messages.length * 8)
        : 0

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
    })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, loading, finalResult])

  const syncConversationParam = (id: number | null) => {
    const current = searchParams.get('conversation')
    const nextValue = id != null ? String(id) : null
    if (current === nextValue) return
    const next = new URLSearchParams(searchParams)
    if (nextValue) next.set('conversation', nextValue)
    else next.delete('conversation')
    setSearchParams(next, { replace: true })
  }

  const handleSelectionChange = (next: string) => {
    setSelection(next)
    onModelChange(resolveChatModel(next, models))
  }

  const resetConversation = () => {
    setMessages([])
    setSessionState(null)
    setFinalResult(null)
    setConversationId(null)
    setInput('')
    setError(null)
    setToastOpen(false)
    setSelection(AUTO_MODEL)
    deepLinkHandled.current = null
    syncConversationParam(null)
    requestAnimationFrame(() => landingInputRef.current?.focus())
  }

  useEffect(() => {
    if (newChatRequest == null) return
    resetConversation()
    // Reset when the header New chat control fires.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [newChatRequest])

  const openConversation = async (id: number) => {
    setError(null)
    try {
      const detail = await plantModelApi.getConversation(id)
      setConversationId(detail.id)
      setMessages(detail.messages)
      setSessionState(detail.session_state)
      setFinalResult(detail.final_result)
      setToastOpen(false)
      setInput('')
      syncConversationParam(detail.id)
      if (detail.llm_model) {
        setSelection(detail.llm_model)
        onModelChange(detail.llm_model)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open conversation')
    }
  }

  useEffect(() => {
    const raw = searchParams.get('conversation')
    if (!raw) return
    if (deepLinkHandled.current === raw) return
    const id = Number(raw)
    if (!Number.isFinite(id) || id <= 0) return
    deepLinkHandled.current = raw
    if (conversationId === id) return
    void openConversation(id)
    // Intentionally only when the query param changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams])

  const sendMessage = async (raw?: string) => {
    const trimmed = (raw ?? input).trim()
    if (!trimmed || chatDisabled) return

    setError(null)
    setLoading(true)
    setInput('')
    if (threadInputRef.current) {
      threadInputRef.current.style.height = 'auto'
    }

    const historyBefore = messages
    const userMessage: PlantModelChatMessage = { role: 'user', content: trimmed }
    setMessages((prev) => [...prev, userMessage])

    try {
      const response = await plantModelApi.chat({
        messages: historyBefore,
        user_message: trimmed,
        model: resolvedModel,
        session_state: sessionState,
        conversation_id: conversationId,
      })

      onModelChange(resolvedModel)

      if (response.conversation_id != null) {
        setConversationId(response.conversation_id)
        deepLinkHandled.current = String(response.conversation_id)
        syncConversationParam(response.conversation_id)
      }
      setMessages((prev) => [...prev, { role: 'assistant', content: response.reply }])
      setSessionState(response.session_state)
      if (response.final_result) {
        setFinalResult(response.final_result)
      } else if (response.status === 'complete' && draft) {
        setFinalResult(draft)
      } else if (response.status !== 'complete' && trimmed !== 'confirm' && trimmed !== 'finish') {
        // Allow revise: clear the locked plant panel until a new complete result arrives.
        setFinalResult(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chat request failed')
      setMessages((prev) => prev.slice(0, -1))
      setInput(trimmed)
    } finally {
      setLoading(false)
      requestAnimationFrame(() => threadInputRef.current?.focus())
    }
  }

  const caseStudiesHref =
    conversationId != null ? `/case-studies?new=${conversationId}` : '/case-studies'

  const handleLaunch = () => {
    if (!finalResult) return
    setToastOpen(false)
    onUseModel(finalResult, conversationId)
  }

  const handleConfirmDraft = async () => {
    if (!draft || loading) return
    const confirmedDraft = draft
    setFinalResult(confirmedDraft)
    await sendMessage('confirm')
  }

  const handleDownload = () => {
    if (!finalResult) return
    triggerBlobDownload(
      new Blob([finalResult.python_code], { type: 'text/x-python' }),
      'dynamics.py',
    )
  }

  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void sendMessage()
    }
  }

  const plantRows: { k: string; v: string }[] = []
  if (draft) {
    plantRows.push({ k: 'System', v: draft.system_name })
    if (sessionState?.draft_count) {
      plantRows.push({ k: 'Drafts', v: String(sessionState.draft_count) })
    }
    plantRows.push({
      k: 'Status',
      v: finalResult ? 'Ready to confirm' : 'Draft in progress',
    })
    plantRows.push({ k: 'Artifact', v: 'dynamics(t, x, u) · Python' })
  }
  const metadataEntries = draft?.metadata ? orderedMetadataEntries(draft.metadata) : []

  return (
    <div className="relative flex min-h-0 min-w-0 flex-1 overflow-x-hidden">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-x-hidden">
        {/* Landing */}
        {!inChat && (
          <div className="relative flex min-h-0 flex-1 flex-col items-center overflow-y-auto px-4 py-8 sm:px-6 sm:py-12">
            {/* Subtle ambient background glow */}
            <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
              <div className="absolute left-1/2 -top-24 -translate-x-1/2 h-72 w-[560px] rounded-full bg-gradient-to-b from-primary/15 via-purple-500/5 to-transparent blur-3xl" />
            </div>

            <div className="relative z-10 my-auto flex w-full max-w-[760px] flex-col items-center text-center">
              {/* Badge */}
              <div className="mb-3.5 inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary/10 px-3 py-0.5 text-xs font-medium text-primary backdrop-blur-md">
                <Sparkles className="size-3 text-primary" />
                <span>Physical System Synthesizer</span>
                <span className="size-1 rounded-full bg-primary/50" />
                <span className="text-[11px] font-mono text-primary/80">AgentPlant AI</span>
              </div>

              {/* Title & Subtitle */}
              <h1 className="mb-2 text-2xl font-bold tracking-tight text-foreground sm:text-3xl lg:text-[34px] leading-tight">
                What physical system do you want to model?
              </h1>
              <p className="mx-auto mb-6 max-w-lg text-[13.5px] leading-relaxed text-muted-text">
                Describe your plant dynamics in plain physics, transfer functions, or differential equations. AgentPlant derives continuous ODEs and prepares hand-off for Control Design.
              </p>

              {error && (
                <div className="mb-4 w-full text-left">
                  <StatusMessage type="error" message={error} />
                </div>
              )}

              {/* Elevated Prompt Box */}
              <div className="w-full rounded-2xl border border-border bg-surface-elevated p-3 sm:p-4 shadow-sm backdrop-blur-xl transition-all duration-200 focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/20 dark:border-white/10 dark:bg-[#11161d]/90">
                <textarea
                  ref={landingInputRef}
                  rows={2}
                  className="max-h-40 w-full resize-none border-none bg-transparent px-2 py-1 font-inherit text-[14.5px] leading-relaxed text-foreground outline-none placeholder:text-muted"
                  placeholder="e.g. A nonlinear magnetic levitation ball with coil inductance, or a buck-boost converter stepping 12V to 48V with inductor ESR..."
                  value={input}
                  disabled={chatDisabled}
                  onChange={(e) => {
                    setInput(e.target.value)
                    autoresize(e.target)
                  }}
                  onKeyDown={onComposerKeyDown}
                />
                <div className="mt-2 flex flex-wrap items-center justify-between gap-3 border-t border-border/60 dark:border-white/5 pt-2.5">
                  <div className="flex items-center gap-2">
                    <ComposerModelPicker
                      models={models}
                      value={selection}
                      onChange={handleSelectionChange}
                      disabled={chatDisabled}
                    />
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="hidden sm:inline text-[11px] font-mono text-muted">
                      ↵ Enter to send
                    </span>
                    <button
                      type="button"
                      className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-primary via-indigo-500 to-[#4a63e0] px-3.5 py-1.5 text-xs font-semibold text-white shadow-md shadow-primary/25 transition-all hover:brightness-110 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
                      disabled={chatDisabled || !input.trim()}
                      aria-label="Synthesize Model"
                      onClick={() => void sendMessage()}
                    >
                      {loading ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : (
                        <>
                          <span>Synthesize</span>
                          <ArrowRight className="size-3.5" />
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {/* Recent Systems Quick Shelf (if any) */}
              {recentConversations.length > 0 && (
                <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs">
                  <span className="text-[11px] text-muted flex items-center gap-1 font-medium">
                    <Clock className="size-3" /> Recent:
                  </span>
                  {recentConversations.slice(0, 3).map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() => void openConversation(c.id)}
                      className="rounded-lg border border-border bg-surface-muted hover:bg-surface-hover hover:border-primary/40 px-2.5 py-1 text-[11.5px] text-foreground transition-colors truncate max-w-[200px] dark:border-white/[0.08] dark:bg-white/[0.02] dark:hover:bg-white/[0.06] dark:text-slate-300"
                    >
                      {c.system_name || c.title || 'Untitled system'}
                    </button>
                  ))}
                  <Link to="/case-studies" className="text-[11.5px] text-primary hover:underline ml-1 font-medium">
                    All case studies →
                  </Link>
                </div>
              )}

              {/* Intermediate Archetype Grid (2x2) */}
              <div className="mt-7 w-full text-left">
                <div className="flex items-center justify-between mb-2.5 px-0.5">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">
                    Or start from an engineering archetype
                  </span>
                  <span className="text-[11px] text-muted font-mono hidden sm:inline">
                    Click to load prompt
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {SYSTEM_ARCHETYPES.map((arch) => (
                    <div
                      key={arch.id}
                      onClick={() => void sendMessage(arch.prompt)}
                      className="group relative flex flex-col justify-between rounded-xl border border-border bg-surface-elevated hover:bg-surface-hover hover:border-primary/40 p-3.5 transition-all duration-200 cursor-pointer overflow-hidden shadow-sm dark:border-white/[0.08] dark:bg-[#11161d]/75 dark:hover:bg-[#161d26]"
                    >
                      <div className="pointer-events-none absolute inset-x-0 top-0 h-[1.5px] bg-gradient-to-r from-transparent via-primary/50 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
                      <div>
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <span className="text-[10.5px] font-medium text-muted">
                            {arch.category}
                          </span>
                          <span className="rounded border border-border bg-surface-muted px-1.5 py-0.5 text-[9.5px] font-mono text-muted-text dark:border-white/[0.06] dark:bg-white/[0.04] dark:text-slate-300">
                            {arch.badge}
                          </span>
                        </div>
                        <h3 className="text-[13px] font-semibold text-foreground group-hover:text-primary transition-colors">
                          {arch.title}
                        </h3>
                        <p className="mt-1 text-[11.5px] text-muted-text line-clamp-1 leading-snug dark:text-slate-400">
                          {arch.desc}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Bottom Subtle Navigation */}
              <div className="mt-6 text-[12px] text-muted">
                Already have benchmark systems?{' '}
                <Link
                  to="/case-studies"
                  className="font-medium text-primary hover:underline"
                >
                  Browse Case Studies →
                </Link>
              </div>
            </div>
          </div>
        )}

        {/* Chat layout */}
        {inChat && (
          <div
            className={`grid min-h-0 min-w-0 flex-1 overflow-x-hidden grid-cols-1 ${
              plantPanelOpen
                ? 'lg:grid-cols-[minmax(0,1fr)_minmax(0,380px)] xl:grid-cols-[minmax(0,1fr)_minmax(0,440px)] 2xl:grid-cols-[minmax(0,1fr)_minmax(0,480px)]'
                : ''
            }`}
          >
            <div
              className={`relative flex min-h-0 min-w-0 flex-col border-border ${
                plantPanelOpen ? 'lg:border-r' : ''
              }`}
            >
              {!plantPanelOpen && (
                <div className="absolute right-3 top-3 z-10 hidden lg:block">
                  <button
                    type="button"
                    className="flex size-8 items-center justify-center rounded-lg border border-border bg-surface-elevated text-muted-text shadow-sm transition-colors hover:bg-surface-hover hover:text-foreground"
                    onClick={() => setPlantPanelOpen(true)}
                    aria-label="Open plant model panel"
                    title="Open plant model panel"
                  >
                    <PanelRightOpen className="size-4" />
                  </button>
                </div>
              )}

              {error && (
                <div className="min-w-0 px-5 pt-3 sm:px-8">
                  <StatusMessage type="error" message={error} />
                </div>
              )}

              <div
                ref={listRef}
                className="flex min-h-0 min-w-0 flex-1 flex-col gap-[18px] overflow-x-hidden overflow-y-auto px-5 py-7 sm:px-8"
                aria-live="polite"
              >
                {messages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className={`flex w-full max-w-[960px] min-w-0 gap-3.5 xl:max-w-[1100px] 2xl:max-w-[1240px] ${
                      message.role === 'user' ? 'flex-row-reverse self-end' : ''
                    }`}
                  >
                    <div
                      className={`mt-0.5 grid size-[26px] shrink-0 place-items-center rounded-lg text-[10px] font-bold ${
                        message.role === 'assistant'
                          ? 'bg-gradient-to-br from-primary to-accent-2 text-white'
                          : 'border border-border bg-surface-muted text-muted-text'
                      }`}
                    >
                      {message.role === 'user' ? initials : 'L'}
                    </div>
                    <div
                      className={`min-w-0 break-words rounded-xl px-3.5 py-2.5 text-[13.5px] leading-relaxed ${
                        message.role === 'assistant'
                          ? 'border border-border bg-surface-elevated text-foreground'
                          : 'border border-[color-mix(in_srgb,var(--app-primary)_28%,transparent)] bg-[color-mix(in_srgb,var(--app-primary)_12%,transparent)] text-foreground'
                      }`}
                    >
                      {message.role === 'assistant' ? (
                        <MarkdownContent content={message.content} className="plant-chat-md" />
                      ) : (
                        <p className="m-0 whitespace-pre-wrap break-words">{message.content}</p>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex w-full max-w-[960px] min-w-0 gap-3.5 xl:max-w-[1100px] 2xl:max-w-[1240px]">
                    <div className="mt-0.5 grid size-[26px] shrink-0 place-items-center rounded-lg bg-gradient-to-br from-primary to-accent-2 text-[10px] font-bold text-white">
                      L
                    </div>
                    <div className="rounded-xl border border-border bg-surface-elevated px-3.5 py-2.5">
                      <div className="flex gap-1 px-0.5 py-1">
                        <span className="size-[5px] animate-bounce rounded-full bg-muted [animation-delay:0ms]" />
                        <span className="size-[5px] animate-bounce rounded-full bg-muted [animation-delay:150ms]" />
                        <span className="size-[5px] animate-bounce rounded-full bg-muted [animation-delay:300ms]" />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="min-w-0 border-t border-border px-5 py-4 sm:px-8 sm:pb-5">
                <div className="flex flex-col gap-2 rounded-2xl border border-border bg-surface-elevated px-3 py-2.5 transition-[border-color] duration-150 focus-within:border-[color-mix(in_srgb,var(--app-primary)_28%,transparent)]">
                  <textarea
                    ref={threadInputRef}
                    rows={1}
                    className="max-h-40 w-full resize-none border-none bg-transparent px-1.5 py-2 font-inherit text-[14.5px] leading-normal text-foreground outline-none placeholder:text-muted disabled:opacity-60"
                    placeholder={
                      finalResult
                        ? 'Conversation complete — start a new chat to design another system.'
                        : 'Reply...'
                    }
                    value={input}
                    disabled={chatDisabled}
                    onChange={(e) => {
                      setInput(e.target.value)
                      autoresize(e.target)
                    }}
                    onKeyDown={onComposerKeyDown}
                  />
                  <div className="flex min-w-0 items-center justify-between gap-2">
                    <ComposerModelPicker
                      models={models}
                      value={selection}
                      onChange={handleSelectionChange}
                      disabled={disabled || loading}
                    />
                    <button
                      type="button"
                      className="grid size-9 shrink-0 place-items-center rounded-[10px] border-none bg-gradient-to-br from-primary to-[#4a63e0] text-white shadow-[0_2px_12px_color-mix(in_srgb,var(--app-primary)_30%,transparent)] transition-[filter] duration-[130ms] hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-35"
                      disabled={chatDisabled || !input.trim()}
                      aria-label="Send"
                      onClick={() => void sendMessage()}
                    >
                      {loading ? (
                        <Loader2 className="size-4 animate-spin" aria-hidden />
                      ) : (
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="size-4"
                          aria-hidden
                        >
                          <path d="M5 12h14M13 6l6 6-6 6" />
                        </svg>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Plant model side panel */}
            {plantPanelOpen && (
              <aside className="hidden min-h-0 min-w-0 flex-col overflow-hidden bg-surface-elevated lg:flex">
                <div className="border-b border-border px-5 pb-3.5 pt-[18px]">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.05em] text-muted">
                        Building
                      </div>
                      <h3 className="m-0 text-[15px] font-semibold text-foreground">Plant model</h3>
                    </div>
                    <button
                      type="button"
                      className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-surface text-muted-text transition-colors hover:bg-surface-hover hover:text-foreground"
                      onClick={() => setPlantPanelOpen(false)}
                      aria-label="Collapse plant model panel"
                      title="Collapse plant model panel"
                    >
                      <PanelRightClose className="size-4" />
                    </button>
                  </div>
                </div>

                <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-x-hidden overflow-y-auto px-5 py-4">
                  {plantRows.length === 0 ? (
                    <p className="m-0 px-2.5 py-[30px] text-center text-[12.5px] leading-relaxed text-muted">
                      As we talk through the system, the model takes shape here — structure, order, and
                      the parameters that matter for control.
                    </p>
                  ) : (
                    plantRows.map((row) => (
                      <div
                        key={row.k}
                        className="flex min-w-0 items-start justify-between gap-3 rounded-[10px] border border-border-subtle bg-surface-muted px-3 py-2.5"
                      >
                        <div className="shrink-0 text-[11.5px] font-semibold text-muted">{row.k}</div>
                        <div className="min-w-0 break-all text-right font-mono text-[12.5px] text-foreground">
                          {row.v}
                        </div>
                      </div>
                    ))
                  )}

                  {metadataEntries.length > 0 && (
                    <>
                      <div className="mt-1 text-[11px] font-semibold uppercase tracking-[0.05em] text-muted">
                        Model details
                      </div>
                      {metadataEntries.map(([key, value]) => (
                        <div
                          key={key}
                          className="min-w-0 rounded-[10px] border border-border-subtle bg-surface-muted px-3 py-2.5"
                        >
                          <div className="mb-1.5 text-[11.5px] font-semibold text-muted">
                            {humanizeMetadataKey(key)}
                          </div>
                          <MetadataValue value={value} />
                        </div>
                      ))}
                    </>
                  )}

                  {draft && (
                    <div className="mt-1 min-w-0 overflow-hidden rounded-[10px] border border-border-subtle">
                      <CodePreview value={draft.python_code} readOnly height={260} language="python" />
                    </div>
                  )}
                </div>

                <div className="px-5 pb-4 pt-1">
                  <div className="h-1.5 overflow-hidden rounded-full bg-surface-muted shadow-inner">
                    <div
                      className="h-full bg-gradient-to-r from-primary via-indigo-400 to-[var(--app-status-success-text)] shadow-[0_0_8px_rgba(99,102,241,0.4)] transition-[width] duration-500"
                      style={{ width: `${progressPct}%` }}
                    />
                  </div>
                  <div
                    className={`mt-1.5 text-[11px] ${
                      progressPct >= 100 ? 'text-[var(--app-status-success-text)]' : 'text-muted'
                    }`}
                  >
                    {progressPct >= 100
                      ? 'Model complete — ready to confirm'
                      : progressPct === 0
                        ? 'Not started'
                        : `${progressPct}% shaped`}
                  </div>
                </div>

                <div className="border-t border-border px-5 pb-5 pt-4">
                  {finalResult && (
                    <button
                      type="button"
                      className={`${btnBase} ${btnCompact} mb-2 w-full justify-center`}
                      onClick={handleDownload}
                    >
                      Download dynamics.py
                    </button>
                  )}
                  {finalResult ? (
                    <button
                      type="button"
                      className={`${btnPrimary} w-full justify-center py-2.5 text-[13.5px]`}
                      disabled={disabled}
                      onClick={handleLaunch}
                    >
                      Save &amp; View in Case Studies →
                    </button>
                  ) : (
                    <button
                      type="button"
                      className={`${btnPrimary} w-full justify-center py-2.5 text-[13.5px]`}
                      disabled={!draft || disabled || loading}
                      onClick={() => void handleConfirmDraft()}
                    >
                      Confirm system
                    </button>
                  )}
                </div>
              </aside>
            )}
          </div>
        )}

        {/* Confirm bar: mobile always; desktop when plant panel is collapsed */}
        {inChat && (finalResult || draft) && (
          <div
            className={`border-t border-border px-4 py-3 ${
              plantPanelOpen ? 'lg:hidden' : ''
            }`}
          >
            {finalResult ? (
              <button
                type="button"
                className={`${btnPrimary} w-full justify-center`}
                disabled={disabled}
                onClick={handleLaunch}
              >
                Save &amp; View in Case Studies →
              </button>
            ) : (
              <button
                type="button"
                className={`${btnPrimary} w-full justify-center`}
                disabled={disabled || loading}
                onClick={() => void handleConfirmDraft()}
              >
                Confirm system
              </button>
            )}
          </div>
        )}
      </div>

      {/* Toast */}
      {!isModalOpen && (
        <div
          className={`fixed bottom-[26px] left-1/2 z-40 flex items-center gap-3.5 rounded-xl border border-border bg-surface-muted px-4 py-3.5 shadow-[0_16px_50px_rgba(0,0,0,0.5)] transition-all duration-250 ${
            toastOpen
              ? 'pointer-events-auto translate-x-[-50%] translate-y-0 opacity-100'
              : 'pointer-events-none translate-x-[-50%] translate-y-5 opacity-0'
          }`}
        >
          <div className="grid size-[30px] shrink-0 place-items-center rounded-lg bg-[var(--app-status-success-bg)] text-[var(--app-status-success-text)]">
            <Check className="size-[15px]" strokeWidth={2.5} aria-hidden />
          </div>
          <div>
            <div className="text-[13px] font-semibold text-foreground">Ready for Case Studies &amp; Control Design</div>
            <div className="text-xs text-muted">{finalResult?.system_name ?? 'Plant model'}</div>
          </div>
          <div className="ml-2 flex gap-2">
            <Link to={caseStudiesHref} className={`${btnBase} ${btnCompact}`}>
              Case Studies
            </Link>
            <button
              type="button"
              className={`${btnPrimary} ${btnCompact}`}
              onClick={handleLaunch}
            >
              Save &amp; Launch →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
