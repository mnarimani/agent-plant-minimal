/**
 * AgentPlant chat shell matching the approved ChatGPT-style mockup
 * (chatgpt-style-ui.html). Wires plant-model chat, upload, and simulate.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Plot from 'react-plotly.js'
import { plantModelApi } from '../api/endpoints'
import type {
  PlantModelChatMessage,
  PlantModelChatResponse,
  PlantModelResult,
  PlantModelSessionState,
  PlantModelStep,
  SimulateResponse,
} from '../api/types'
import { MarkdownContent } from './MarkdownContent'
import { CodePreview } from './CodePreview'
import './agentplant-chat.css'

type PanelView = 'code' | 'sandbox' | null

type Turn = {
  id: string
  role: 'user' | 'assistant'
  content: string
  status?: 'continue' | 'draft' | 'complete'
  steps?: PlantModelStep[]
  fileName?: string
  at: number
}

/** Relative time like the mockup (“24 minutes ago”), not wall-clock. */
function formatRelativeTime(at: number, now: number = Date.now()): string {
  const sec = Math.max(0, Math.floor((now - at) / 1000))
  if (sec < 10) return 'just now'
  if (sec < 60) return `${sec} seconds ago`
  const min = Math.floor(sec / 60)
  if (min === 1) return '1 minute ago'
  if (min < 60) return `${min} minutes ago`
  const hr = Math.floor(min / 60)
  if (hr === 1) return '1 hour ago'
  if (hr < 24) return `${hr} hours ago`
  const day = Math.floor(hr / 24)
  if (day === 1) return '1 day ago'
  if (day < 30) return `${day} days ago`
  const mo = Math.floor(day / 30)
  if (mo === 1) return '1 month ago'
  return `${mo} months ago`
}

const IconCopy = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <rect x="9" y="9" width="13" height="13" rx="2" />
    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
  </svg>
)
const IconRegen = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <path d="M1 4v6h6M23 20v-6h-6" />
    <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" />
  </svg>
)
const IconEdit = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <path d="M12 20h9M16.5 3.5a2.12 2.12 0 013 3L7 19l-4 1 1-4L16.5 3.5z" />
  </svg>
)
const IconLike = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14zM7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3" />
  </svg>
)
const IconDislike = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10zM17 2h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17" />
  </svg>
)
const IconDots = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <circle cx="5" cy="12" r="1.75" />
    <circle cx="12" cy="12" r="1.75" />
    <circle cx="19" cy="12" r="1.75" />
  </svg>
)

type Feedback = 'up' | 'down' | null

type TurnMetaProps = {
  turn: Turn
  now: number
  feedback: Feedback
  menuOpen: boolean
  onCopy: () => void
  onRegen?: () => void
  onEdit?: () => void
  onLike?: () => void
  onDislike?: () => void
  onMenuToggle: () => void
  onMenuClose: () => void
}

function TurnMeta({
  turn,
  now,
  feedback,
  menuOpen,
  onCopy,
  onRegen,
  onEdit,
  onLike,
  onDislike,
  onMenuToggle,
  onMenuClose,
}: TurnMetaProps) {
  const isUser = turn.role === 'user'
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!menuOpen) return
    const onDoc = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) onMenuClose()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onMenuClose()
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [menuOpen, onMenuClose])

  return (
    <div className={`ap-meta${isUser ? ' ap-meta--user' : ' ap-meta--assistant'}`}>
      {isUser && <span className="ap-meta__time">{formatRelativeTime(turn.at, now)}</span>}
      <div className="ap-meta__actions">
        <button type="button" className="ap-meta__btn" aria-label="Copy" title="Copy" onClick={onCopy}>
          <IconCopy />
        </button>
        {isUser && onEdit && (
          <button type="button" className="ap-meta__btn" aria-label="Edit" title="Edit" onClick={onEdit}>
            <IconEdit />
          </button>
        )}
        {!isUser && (
          <>
            <button
              type="button"
              className={`ap-meta__btn${feedback === 'up' ? ' ap-meta__btn--active' : ''}`}
              aria-label="Good response"
              title="Good response"
              aria-pressed={feedback === 'up'}
              onClick={onLike}
            >
              <IconLike />
            </button>
            <button
              type="button"
              className={`ap-meta__btn${feedback === 'down' ? ' ap-meta__btn--active' : ''}`}
              aria-label="Bad response"
              title="Bad response"
              aria-pressed={feedback === 'down'}
              onClick={onDislike}
            >
              <IconDislike />
            </button>
          </>
        )}
        {onRegen && (
          <button type="button" className="ap-meta__btn" aria-label="Regenerate" title="Regenerate" onClick={onRegen}>
            <IconRegen />
          </button>
        )}
        <div className="ap-meta__more" ref={menuRef}>
          <button
            type="button"
            className={`ap-meta__btn${menuOpen ? ' ap-meta__btn--active' : ''}`}
            aria-label="More options"
            title="More"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            onClick={onMenuToggle}
          >
            <IconDots />
          </button>
          {menuOpen && (
            <div className="ap-meta__menu" role="menu">
              <button type="button" role="menuitem" className="ap-meta__menu-item" disabled title="Coming soon">
                Report issue
              </button>
              <button type="button" role="menuitem" className="ap-meta__menu-item" disabled title="Coming soon">
                Generate PDF
              </button>
            </div>
          )}
        </div>
      </div>
      {!isUser && <span className="ap-meta__time">{formatRelativeTime(turn.at, now)}</span>}
    </div>
  )
}

const DEMO_CODE = `import numpy as np

def dynamics(t, x, u):
    """Quanser-style 2-DOF plant (demo)."""
    theta, psi, theta_d, psi_d = x
    Vp, Vy = u[0], u[1] if len(u) > 1 else 0.0
    Jp, Jy = 0.0384, 0.0432
    Kpp, Kyy = 0.204, 0.072
    Bp, By = 0.8, 0.318
    theta_dd = (Kpp * Vp - Bp * theta_d) / Jp
    psi_dd = (Kyy * Vy - By * psi_d) / Jy
    return [theta_d, psi_d, theta_dd, psi_dd]
`

type Props = {
  model?: string
  onUseModel?: (result: PlantModelResult, conversationId?: number | null) => void
}

export function AgentPlantChat({ model = "gpt-4o-mini", onUseModel }: Props) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [searchOn, setSearchOn] = useState(false)
  const [attached, setAttached] = useState<string[]>([])
  const [session, setSession] = useState<PlantModelSessionState | null>(null)
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [draft, setDraft] = useState<PlantModelResult | null>(null)
  const [panel, setPanel] = useState<PanelView>(null)
  const [traceOpen, setTraceOpen] = useState(false)
  const [labOpen, setLabOpen] = useState(false)
  const [sim, setSim] = useState<SimulateResponse | null>(null)
  const [simLoading, setSimLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Lab controls
  const [channel, setChannel] = useState(0)
  const [wave, setWave] = useState('step')
  const [T, setT] = useState(10)
  const [t0, setT0] = useState(1)
  const [amp, setAmp] = useState(1)
  const fileRef = useRef<HTMLInputElement>(null)
  const [now, setNow] = useState(() => Date.now())
  const [feedbackById, setFeedbackById] = useState<Record<string, Feedback>>({})
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null)
  /** Inline edit of an existing user turn (same message id, not a new composer send). */
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editDraft, setEditDraft] = useState('')

  // Keep relative timestamps fresh while the chat is open
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 30_000)
    return () => window.clearInterval(id)
  }, [])

  const messages: PlantModelChatMessage[] = useMemo(
    () => turns.map((t) => ({ role: t.role, content: t.content })),
    [turns],
  )

  const copyText = useCallback(async (text: string) => {
    try {
      await navigator.clipboard?.writeText(text)
    } catch {
      /* ignore */
    }
  }, [])

  const startEditUserTurn = useCallback((turn: Turn) => {
    setEditingId(turn.id)
    setEditDraft(turn.content)
    setMenuOpenId(null)
  }, [])

  const cancelEditUserTurn = useCallback(() => {
    setEditingId(null)
    setEditDraft('')
  }, [])

  /** Apply edit on the same user bubble, drop later turns, re-run the agent. */
  const saveEditUserTurn = useCallback(
    async (turnId: string) => {
      const trimmed = editDraft.trim()
      if (!trimmed || loading) return
      const idx = turns.findIndex((t) => t.id === turnId)
      if (idx < 0) return

      const hist = turns
        .slice(0, idx)
        .map((t) => ({ role: t.role, content: t.content }) as PlantModelChatMessage)
      const prev = turns[idx]
      const updatedUser: Turn = {
        ...prev,
        content: trimmed,
        at: Date.now(),
      }

      setEditingId(null)
      setEditDraft('')
      setError(null)
      setDraft(null)
      // Keep this user message; remove everything after it
      setTurns((all) => [...all.slice(0, idx), updatedUser])
      setLoading(true)

      try {
        const res: PlantModelChatResponse = await plantModelApi.chat({
          messages: hist,
          user_message: trimmed,
          model,
          session_state: session,
          conversation_id: conversationId,
          web_search_enabled: searchOn,
        })
        setSession(res.session_state)
        if (res.conversation_id != null) setConversationId(res.conversation_id)
        const nextDraft = res.draft ?? res.final_result ?? res.session_state?.latest_draft ?? null
        if (nextDraft) {
          setDraft(nextDraft)
          setPanel((p) => p ?? 'code')
        }
        const nextSteps = res.steps ?? []
        if (nextSteps.some((s) => s.kind === 'rag' || s.kind === 'web')) setTraceOpen(true)
        setTurns((all) => [
          ...all,
          {
            id: `a-${Date.now()}`,
            role: 'assistant',
            content: res.reply,
            status: res.status,
            steps: nextSteps,
            at: Date.now(),
          },
        ])
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e)
        setError(msg)
        setTurns((all) => [
          ...all,
          {
            id: `a-err-${Date.now()}`,
            role: 'assistant',
            content: `Error: ${msg}`,
            status: 'continue',
            at: Date.now(),
          },
        ])
      } finally {
        setLoading(false)
      }
    },
    [editDraft, loading, turns, model, session, conversationId, searchOn],
  )

  const resendMessage = useCallback(
    async (text: string, historyOverride?: PlantModelChatMessage[]) => {
      const trimmed = text.trim()
      if (!trimmed || loading) return
      setError(null)
      setInput('')
      const userTurn: Turn = {
        id: `u-${Date.now()}`,
        role: 'user',
        content: trimmed,
        fileName: attached[0],
        at: Date.now(),
      }
      setTurns((prev) => [...prev, userTurn])
      setLoading(true)
      try {
        const hist =
          historyOverride ??
          turns.map((t) => ({ role: t.role, content: t.content }) as PlantModelChatMessage)
        const res: PlantModelChatResponse = await plantModelApi.chat({
          messages: hist,
          user_message: trimmed,
          model,
          session_state: session,
          conversation_id: conversationId,
          web_search_enabled: searchOn,
        })
        setSession(res.session_state)
        if (res.conversation_id != null) setConversationId(res.conversation_id)
        const nextDraft = res.draft ?? res.final_result ?? res.session_state?.latest_draft ?? null
        if (nextDraft) {
          setDraft(nextDraft)
          setPanel((p) => p ?? 'code')
        }
        const nextSteps = res.steps ?? []
        if (nextSteps.some((s) => s.kind === 'rag' || s.kind === 'web')) setTraceOpen(true)
        setTurns((prev) => [
          ...prev,
          {
            id: `a-${Date.now()}`,
            role: 'assistant',
            content: res.reply,
            status: res.status,
            steps: nextSteps,
            at: Date.now(),
          },
        ])
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e)
        setError(msg)
        setTurns((prev) => [
          ...prev,
          {
            id: `a-err-${Date.now()}`,
            role: 'assistant',
            content: `Error: ${msg}`,
            status: 'continue',
            at: Date.now(),
          },
        ])
      } finally {
        setLoading(false)
      }
    },
    [loading, attached, turns, model, session, conversationId, searchOn],
  )

  const handleRegenUser = useCallback(
    (turn: Turn) => {
      const idx = turns.findIndex((t) => t.id === turn.id)
      const hist = turns.slice(0, idx).map((t) => ({ role: t.role, content: t.content }))
      setTurns((prev) => prev.slice(0, idx))
      setDraft(null)
      void resendMessage(turn.content, hist)
    },
    [turns, resendMessage],
  )

  const handleRegenAssistant = useCallback(
    (turn: Turn) => {
      // Re-run using the nearest preceding user message
      const idx = turns.findIndex((t) => t.id === turn.id)
      if (idx < 0) return
      let userIdx = -1
      for (let i = idx - 1; i >= 0; i--) {
        if (turns[i].role === 'user') {
          userIdx = i
          break
        }
      }
      if (userIdx < 0) return
      const userTurn = turns[userIdx]
      const hist = turns.slice(0, userIdx).map((t) => ({ role: t.role, content: t.content }))
      setTurns((prev) => prev.slice(0, userIdx))
      setDraft(null)
      void resendMessage(userTurn.content, hist)
    },
    [turns, resendMessage],
  )

  const send = useCallback(async () => {
    const text = input.trim()
    if (!text || loading) return
    setError(null)
    setInput('')
    const userTurn: Turn = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: text,
      fileName: attached[0],
      at: Date.now(),
    }
    setTurns((prev) => [...prev, userTurn])
    setLoading(true)
    try {
      const res: PlantModelChatResponse = await plantModelApi.chat({
        messages,
        user_message: text,
        model,
        session_state: session,
        conversation_id: conversationId,
        web_search_enabled: searchOn,
      })
      setSession(res.session_state)
      if (res.conversation_id != null) setConversationId(res.conversation_id)
      const nextDraft = res.draft ?? res.final_result ?? res.session_state?.latest_draft ?? null
      if (nextDraft) {
        setDraft(nextDraft)
        setPanel((p) => p ?? 'code')
      }
      const nextSteps = res.steps ?? []
      if (nextSteps.some((s) => s.kind === 'rag' || s.kind === 'web')) setTraceOpen(true)
      setTurns((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: res.reply,
          status: res.status,
          steps: nextSteps,
          at: Date.now(),
        },
      ])
      if (res.status === 'complete' && nextDraft && onUseModel) {
        // expose complete for pipeline handoff without forcing navigation
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      setTurns((prev) => [
        ...prev,
        {
          id: `a-err-${Date.now()}`,
          role: 'assistant',
          content: `Error: ${msg}`,
          status: 'continue',
          at: Date.now(),
        },
      ])
    } finally {
      setLoading(false)
    }
  }, [input, loading, messages, model, session, conversationId, searchOn, attached, onUseModel])

  const onUpload = async (file: File) => {
    setError(null)
    try {
      const res = await plantModelApi.upload(file)
      setAttached(res.attached_files)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const runSim = async () => {
    const code = draft?.python_code || DEMO_CODE
    setSimLoading(true)
    setError(null)
    try {
      const res = await plantModelApi.simulate({
        python_code: code,
        T,
        dt: 0.01,
        input_kind: wave,
        input_channel: channel,
        amplitude: amp,
        t0,
      })
      setSim(res)
      setLabOpen(true)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSimLoading(false)
    }
  }

  const openCode = () => setPanel('code')
  const openLab = () => {
    setPanel('sandbox')
    if (!sim && draft) void runSim()
  }

  const plotData = useMemo(() => {
    if (!sim?.t || !sim?.x) return null
    const traces = (sim.x[0] ? sim.x[0].map((_, si) => si) : []).map((si) => ({
      x: sim.t!,
      y: sim.x!.map((row) => row[si]),
      name: sim.state_names?.[si] ?? `x${si}`,
      mode: 'lines' as const,
      line: { width: 1.5 },
    }))
    if (sim.u) {
      traces.push({
        x: sim.t!,
        y: sim.u.map((row) => row[0] ?? 0),
        name: `u[${sim.input_channel}]`,
        mode: 'lines' as const,
        line: { width: 1.5, dash: 'dash' } as unknown as { width: number },
      })
    }
    return traces
  }, [sim])

  const statusClass =
    sim && !sim.success ? 'ap-lab-status--err' : sim?.diverged ? 'ap-lab-status--warn' : ''

  return (
    <div className={`ap-shell${panel ? '' : ' panel-closed'}`}>
      <section className="ap-chat" aria-label="Conversation">
        <div className="ap-chat__scroll">
          <div className="ap-chat__inner">
            {turns.length === 0 && (
              <div className="ap-empty">
                <h2>Describe a plant model</h2>
                <p>
                  Start with the physical system — states, inputs, parameters. AgentPlant drafts
                  dynamics code and can simulate it in the sandbox.
                </p>
              </div>
            )}

            {turns.map((t) =>
              t.role === 'user' ? (
                <div className="ap-user" key={t.id}>
                  {t.fileName && <div className="ap-user__file">PDF/MD · {t.fileName}</div>}
                  {editingId === t.id ? (
                    <div className="ap-user__edit">
                      <textarea
                        className="ap-user__edit-input"
                        value={editDraft}
                        rows={Math.min(8, Math.max(2, editDraft.split('\n').length))}
                        aria-label="Edit message"
                        autoFocus
                        disabled={loading}
                        onChange={(e) => setEditDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Escape') {
                            e.preventDefault()
                            cancelEditUserTurn()
                          }
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            void saveEditUserTurn(t.id)
                          }
                        }}
                      />
                      <div className="ap-user__edit-actions">
                        <button
                          type="button"
                          className="ap-user__edit-btn ap-user__edit-btn--ghost"
                          disabled={loading}
                          onClick={cancelEditUserTurn}
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          className="ap-user__edit-btn ap-user__edit-btn--primary"
                          disabled={loading || !editDraft.trim()}
                          onClick={() => void saveEditUserTurn(t.id)}
                        >
                          {loading ? 'Sending…' : 'Save'}
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="ap-user__bubble">{t.content}</div>
                      <TurnMeta
                        turn={t}
                        now={now}
                        feedback={null}
                        menuOpen={menuOpenId === t.id}
                        onCopy={() => void copyText(t.content)}
                        onEdit={() => startEditUserTurn(t)}
                        onRegen={() => handleRegenUser(t)}
                        onMenuToggle={() => setMenuOpenId((id) => (id === t.id ? null : t.id))}
                        onMenuClose={() => setMenuOpenId(null)}
                      />
                    </>
                  )}
                </div>
              ) : (
                <div className="ap-assistant" key={t.id}>
                  {!!t.steps?.length && (
                    <div>
                      <button
                        type="button"
                        className="ap-trace-summary"
                        aria-expanded={traceOpen}
                        onClick={() => setTraceOpen((v) => !v)}
                      >
                        {t.steps.length} step{t.steps.length === 1 ? '' : 's'}
                        <span aria-hidden>{traceOpen ? '▴' : '▾'}</span>
                      </button>
                      {traceOpen && (
                        <div className="ap-trace-list" role="region" aria-label="Step log">
                          {t.steps.map((s, i) => (
                            <div className="ap-trace-item" key={`${t.id}-s-${i}`}>
                              <strong>{s.label}</strong>
                              {s.detail && (
                                <div style={{ fontSize: 12.5, marginTop: 4, color: '#8e8e8e' }}>
                                  {s.detail}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="ap-assistant__bubble">
                    {t.status && (
                      <span className={`ap-pill ap-pill--${t.status}`}>{t.status}</span>
                    )}
                    <MarkdownContent content={t.content} className="ap-md" />
                  </div>

                  <TurnMeta
                    turn={t}
                    now={now}
                    feedback={feedbackById[t.id] ?? null}
                    menuOpen={menuOpenId === t.id}
                    onCopy={() => void copyText(t.content)}
                    onRegen={() => handleRegenAssistant(t)}
                    onLike={() =>
                      setFeedbackById((m) => ({
                        ...m,
                        [t.id]: m[t.id] === 'up' ? null : 'up',
                      }))
                    }
                    onDislike={() =>
                      setFeedbackById((m) => ({
                        ...m,
                        [t.id]: m[t.id] === 'down' ? null : 'down',
                      }))
                    }
                    onMenuToggle={() => setMenuOpenId((id) => (id === t.id ? null : t.id))}
                    onMenuClose={() => setMenuOpenId(null)}
                  />
                </div>
              ),
            )}

            {(draft || loading) && (
              <div className="ap-artifacts">
                {draft && (
                  <button
                    type="button"
                    className="ap-artifact"
                    aria-pressed={panel === 'code'}
                    onClick={openCode}
                  >
                    <div className="ap-artifact__icon ap-artifact__icon--code" aria-hidden>
                      {'</>'}
                    </div>
                    <div className="ap-artifact__body">
                      <div className="ap-artifact__title">{draft.system_name || 'dynamics'}</div>
                      <div className="ap-artifact__type">PY</div>
                    </div>
                    <span className="ap-artifact__action">Open</span>
                  </button>
                )}
                {draft && (
                  <button
                    type="button"
                    className="ap-artifact"
                    aria-pressed={panel === 'sandbox'}
                    onClick={openLab}
                  >
                    <div className="ap-artifact__icon ap-artifact__icon--lab" aria-hidden>
                      ⌬
                    </div>
                    <div className="ap-artifact__body">
                      <div className="ap-artifact__title">Sandbox</div>
                      <div className="ap-artifact__type">Artifact</div>
                    </div>
                    <span className="ap-artifact__action">Open</span>
                  </button>
                )}
              </div>
            )}

            {error && (
              <div style={{ color: '#f0576b', fontSize: 13 }} role="alert">
                {error}
              </div>
            )}
          </div>
        </div>

        <div className="ap-composer-wrap">
          <div className="ap-composer">
            <textarea
              rows={1}
              placeholder="Describe the plant…"
              value={input}
              aria-label="Message"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  void send()
                }
              }}
            />
            <div className="ap-composer__tools">
              <button
                type="button"
                className="ap-tool"
                aria-pressed={searchOn}
                aria-label={searchOn ? 'Web search on' : 'Web search off'}
                title="Web search"
                onClick={() => setSearchOn((v) => !v)}
              >
                ◎
              </button>
              <button
                type="button"
                className={`ap-tool${attached.length ? ' ap-tool--attached' : ''}`}
                aria-label={attached[0] ? `Attached: ${attached[0]}` : 'Attach PDF or Markdown'}
                title={attached[0] || 'Attach PDF or Markdown'}
                onClick={() => fileRef.current?.click()}
              >
                📎
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,.md,.txt,application/pdf,text/markdown,text/plain"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) void onUpload(f)
                  e.target.value = ''
                }}
              />
              <button
                type="button"
                className="ap-send"
                aria-label="Send"
                disabled={!input.trim() || loading}
                onClick={() => void send()}
              >
                ↑
              </button>
            </div>
          </div>
        </div>
      </section>

      <aside className="ap-panel" aria-label="Artifact panel">
        <header className="ap-panel__header">
          <div className="ap-panel__title">
            {panel === 'sandbox'
              ? 'Sandbox · Artifact'
              : `${draft?.system_name || 'dynamics'} · PY`}
          </div>
          <button
            type="button"
            className="ap-panel__btn"
            onClick={() => {
              const text =
                panel === 'sandbox'
                  ? JSON.stringify(sim, null, 2)
                  : draft?.python_code || DEMO_CODE
              void navigator.clipboard?.writeText(text || '')
            }}
          >
            Copy
          </button>
          <button type="button" className="ap-panel__btn" onClick={() => setPanel(null)} aria-label="Close">
            ×
          </button>
        </header>

        <div className="ap-panel__body">
          {panel === 'code' && (
            <CodePreview
              value={draft?.python_code || DEMO_CODE}
              readOnly
              height={560}
              language="python"
              showLineNumbers
              className="ap-code-preview"
            />
          )}

          {panel === 'sandbox' && (
            <div className="ap-lab">
              <div className="ap-lab-card">
                <button
                  type="button"
                  className="ap-lab-toggle"
                  aria-expanded={labOpen}
                  onClick={() => setLabOpen((v) => !v)}
                >
                  <span className="dot" aria-hidden />
                  Simulation Lab
                  <span className="chev" aria-hidden>
                    ▾
                  </span>
                </button>
                {labOpen && (
                  <div className="ap-lab-body">
                    <div className="ap-lab-grid">
                      <div className="ap-field">
                        <label htmlFor="ap-ch">Channel</label>
                        <input
                          id="ap-ch"
                          type="number"
                          value={channel}
                          min={0}
                          onChange={(e) => setChannel(Number(e.target.value))}
                        />
                      </div>
                      <div className="ap-field">
                        <label htmlFor="ap-wave">Wave</label>
                        <select
                          id="ap-wave"
                          value={wave}
                          onChange={(e) => setWave(e.target.value)}
                        >
                          <option value="step">step</option>
                          <option value="ramp">ramp</option>
                          <option value="sine">sine</option>
                        </select>
                      </div>
                      <div className="ap-field">
                        <label htmlFor="ap-T">T (s)</label>
                        <input
                          id="ap-T"
                          type="number"
                          value={T}
                          step={0.5}
                          min={0.1}
                          onChange={(e) => setT(Number(e.target.value))}
                        />
                      </div>
                      <div className="ap-field">
                        <label htmlFor="ap-t0">t0 (s)</label>
                        <input
                          id="ap-t0"
                          type="number"
                          value={t0}
                          step={0.1}
                          min={0}
                          onChange={(e) => setT0(Number(e.target.value))}
                        />
                      </div>
                      <div className="ap-field">
                        <label htmlFor="ap-amp">Amp</label>
                        <input
                          id="ap-amp"
                          type="number"
                          value={amp}
                          step={0.1}
                          onChange={(e) => setAmp(Number(e.target.value))}
                        />
                      </div>
                    </div>
                    <button
                      type="button"
                      className="ap-run"
                      disabled={simLoading}
                      onClick={() => void runSim()}
                    >
                      {simLoading ? 'Running…' : 'Run'}
                    </button>
                  </div>
                )}
              </div>

              <div className={`ap-lab-status ${statusClass}`}>
                <span className="dot" aria-hidden />
                {sim ? (
                  <span>
                    <code>{sim.input_kind || wave}</code> · <code>u[{sim.input_channel}]</code>
                    {sim.diverged && ' · diverged'}
                    {!sim.success && ` · ${sim.message.slice(0, 80)}`}
                    {sim.success && !sim.diverged && ` · t ∈ [0, ${T.toFixed(2)}] s`}
                    {sim.demo && ' · demo'}
                  </span>
                ) : (
                  <span>No run yet</span>
                )}
              </div>

              <div className="ap-plot">
                {plotData ? (
                  <Plot
                    data={plotData as object[]}
                    layout={{
                      paper_bgcolor: '#1e1e1e',
                      plot_bgcolor: '#1a1a1a',
                      font: { color: '#8e8e8e', size: 11 },
                      margin: { t: 24, r: 16, b: 40, l: 48 },
                      legend: { orientation: 'h', y: 1.12 },
                      xaxis: { title: 't (s)', gridcolor: '#2a2a2a' },
                      yaxis: { gridcolor: '#2a2a2a' },
                      height: 320,
                      autosize: true,
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                    useResizeHandler
                  />
                ) : (
                  <div style={{ padding: 24, color: '#6b6b6b', fontSize: 13 }}>
                    Run a simulation to see trajectories.
                  </div>
                )}
              </div>

              {sim?.success && sim.t && (
                <button
                  type="button"
                  className="ap-panel__btn"
                  onClick={() => {
                    const rows = ['t,' + (sim.state_names || []).join(',')]
                    sim.t!.forEach((ti, i) => {
                      rows.push([ti, ...(sim.x?.[i] || [])].join(','))
                    })
                    const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
                    const a = document.createElement('a')
                    a.href = URL.createObjectURL(blob)
                    a.download = 'sim.csv'
                    a.click()
                  }}
                >
                  Download CSV
                </button>
              )}
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}
