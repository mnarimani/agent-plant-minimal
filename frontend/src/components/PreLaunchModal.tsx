import { useMemo, useState, useEffect } from 'react'
import { AlertCircle, Loader2, Sparkles, X } from 'lucide-react'
import { plantArtifactApi } from '../api/endpoints'
import type { PlantPayload, PreLaunchConfig } from '../api/types'
import { btnBase, btnCompact, btnPrimary, fieldInput, fieldLabel } from '../lib/classes'

interface PreLaunchModalProps {
  isOpen: boolean
  onClose: () => void
  plant?: PlantPayload | null
  conversationId?: number | null
  systemName?: string
  initialConfig?: Partial<PreLaunchConfig>
  targetPipeline?: 'adaptiveDesign' | 'mpcDesign' | 'siloDesign' | 'muloDesign' | null
  onSuccess: (artifactId: string, preLaunch: PreLaunchConfig) => void
}

function inferNumStates(plant?: PlantPayload | null): number {
  if (Array.isArray(plant?.metadata?.states) && plant.metadata.states.length > 0) {
    return plant.metadata.states.length
  }
  const code = plant?.python_code || ''
  if (!code) return 2

  const indices: number[] = []
  const regex = /(?:x|state|dx|dxdt)\[(\d+)\]/g
  let match: RegExpExecArray | null
  while ((match = regex.exec(code)) !== null) {
    indices.push(parseInt(match[1], 10))
  }
  if (indices.length > 0) {
    return Math.max(...indices) + 1
  }

  const zerosMatch = /np\.(?:zeros|empty|array)\s*\(\s*(\d+)/.exec(code)
  if (zerosMatch) {
    return parseInt(zerosMatch[1], 10)
  }

  return 2
}

function getStateNames(plant: PlantPayload | null | undefined, n: number): string[] {
  const raw = plant?.metadata?.states
  if (Array.isArray(raw) && raw.length > 0) {
    return raw.map((s, i) => (typeof s === 'string' && s.trim() ? s : `x${i + 1}`))
  }
  return Array.from({ length: n }, (_, i) => `x${i + 1}`)
}

function getInputNames(plant?: PlantPayload | null): string[] {
  const raw = plant?.metadata?.inputs
  if (Array.isArray(raw) && raw.length > 0) {
    return raw.map((s, i) => (typeof s === 'string' && s.trim() ? s : `u${i + 1}`))
  }
  return []
}

export function PreLaunchModal({
  isOpen,
  onClose,
  plant,
  conversationId,
  systemName = 'Plant System',
  initialConfig,
  targetPipeline,
  onSuccess,
}: PreLaunchModalProps) {
  const [tSim, setTSim] = useState(initialConfig?.total_simulation_time ?? 10)
  const [dt, setDt] = useState(initialConfig?.solver_sample_time ?? 0.01)

  const detectedStates = useMemo(() => inferNumStates(plant), [plant])
  const stateNames = useMemo(
    () => getStateNames(plant, detectedStates),
    [plant, detectedStates],
  )
  const inputNames = useMemo(() => getInputNames(plant), [plant])

  const defaultX0 = useMemo(() => {
    if (initialConfig?.initial_state?.length === detectedStates) {
      return initialConfig.initial_state.map(Number)
    }
    return new Array(detectedStates).fill(0) as number[]
  }, [initialConfig, detectedStates])

  const [x0Values, setX0Values] = useState<number[]>(defaultX0)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [warnings, setWarnings] = useState<string[]>([])

  useEffect(() => {
    if (isOpen) {
      setX0Values(defaultX0)
      setError(null)
      setWarnings([])
    }
  }, [isOpen, defaultX0])

  if (!isOpen) return null

  const updateX0At = (index: number, raw: string) => {
    const n = Number(raw)
    setX0Values((prev) => {
      const next = [...prev]
      next[index] = Number.isFinite(n) ? n : 0
      return next
    })
  }

  const handleSave = async () => {
    setError(null)
    setWarnings([])

    let parsedX0 = x0Values.map((v) => (Number.isFinite(v) ? v : 0))
    if (parsedX0.length !== detectedStates) {
      if (parsedX0.length === 0 || parsedX0.every((v) => v === 0)) {
        parsedX0 = new Array(detectedStates).fill(0)
      } else {
        setError(`Initial state must have ${detectedStates} elements (currently has ${parsedX0.length})`)
        return
      }
    }

    // Downstream modules set their own targets from the reference trajectory.
    // Do not collect or require a target on the pre-launch form.
    const preLaunch: PreLaunchConfig = {
      total_simulation_time: Number(tSim),
      solver_sample_time: Number(dt),
      initial_state: parsedX0,
      default_target: [],
    }

    setSubmitting(true)
    try {
      const res = await plantArtifactApi.createArtifact({
        pre_launch: preLaunch,
        plant: plant ?? undefined,
        conversation_id: conversationId ?? undefined,
      })
      if (res.warnings?.length) {
        setWarnings(res.warnings)
      }
      onSuccess(res.artifact_id, preLaunch)
      onClose()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to create plant artifact'
      setError(msg)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 dark:bg-black/80 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
    >
      <div className="flex max-h-[90vh] w-full max-w-2xl flex-col rounded-2xl border border-border bg-surface-elevated text-foreground shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="flex size-8 items-center justify-center rounded-lg bg-cyan-500/10 text-cyan-500 dark:text-cyan-400">
              <Sparkles className="size-4" />
            </span>
            <div>
              <h2 className="text-base font-semibold text-foreground">
                Pre-Launch Configuration: {systemName}
                {targetPipeline && (
                  <span className="ml-2 text-xs font-normal text-cyan-600 dark:text-cyan-300">
                    ({targetPipeline === 'adaptiveDesign' ? 'Adaptive' : targetPipeline === 'mpcDesign' ? 'MPC' : 'PID'})
                  </span>
                )}
              </h2>
              <p className="text-xs text-muted">
                Module-agnostic simulation settings & artifact compilation
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-muted hover:bg-surface-muted hover:text-foreground transition-colors"
          >
            <X className="size-4" />
          </button>
        </div>

        {/* Body */}
        <div className="overflow-y-auto px-6 py-5 space-y-5">
          {error && (
            <div className="flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-600 dark:text-rose-300">
              <AlertCircle className="size-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
          {warnings.length > 0 && (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-700 dark:text-amber-300 space-y-1">
              {warnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className={fieldLabel}>Total Simulation Time (s)</label>
              <input
                type="number"
                min={0.1}
                step={0.1}
                value={tSim}
                onChange={(e) => setTSim(Number(e.target.value))}
                className={fieldInput}
              />
              <span className="text-[11px] text-muted">Horizon length for closed-loop simulation</span>
            </div>
            <div className="space-y-1">
              <label className={fieldLabel}>Solver Sample Time dt (s)</label>
              <input
                type="number"
                min={0.0001}
                step={0.001}
                value={dt}
                onChange={(e) => setDt(Number(e.target.value))}
                className={fieldInput}
              />
              <span className="text-[11px] text-muted">Step size for RK4 / OSQP solver</span>
            </div>

            <div className="sm:col-span-2 space-y-2">
              <div className="flex items-center justify-between">
                <label className={fieldLabel}>Initial State x0</label>
                <span className="text-[11px] font-mono text-cyan-600 dark:text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-md border border-cyan-500/20">
                  {detectedStates} states
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {stateNames.map((name, i) => (
                  <div key={`${name}-${i}`} className="space-y-1">
                    <label className="text-[11px] font-medium text-muted">
                      {name}
                      {Array.isArray(plant?.metadata?.state_meanings) &&
                      typeof plant.metadata.state_meanings[i] === 'string' &&
                      plant.metadata.state_meanings[i]
                        ? ` — ${plant.metadata.state_meanings[i]}`
                        : ''}
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={x0Values[i] ?? 0}
                      onChange={(e) => updateX0At(i, e.target.value)}
                      className={`${fieldInput} font-mono text-xs`}
                    />
                  </div>
                ))}
              </div>
              <span className="text-[11px] text-muted">
                Initial condition for each state at t=0
                {inputNames.length > 0 ? ` · Inputs: ${inputNames.join(', ')}` : ''}
              </span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 border-t border-border px-6 py-4">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className={`${btnBase} ${btnCompact}`}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={submitting}
            className={`${btnBase} ${btnCompact} ${btnPrimary} flex items-center gap-2 border-none font-semibold text-white shadow-md`}
          >
            {submitting ? (
              <>
                <Loader2 className="size-3.5 animate-spin" />
                Compiling Artifact...
              </>
            ) : (
              <>Save & Proceed →</>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
