import { useEffect, useId, useRef, useState } from 'react'
import { Check, ChevronDown, Sparkles, Zap } from 'lucide-react'
import {
  AUTO_MODEL,
  modelLabel,
  pickCheapestModel,
} from '../lib/modelPicker'

interface ComposerModelPickerProps {
  models: string[]
  value: string
  onChange: (model: string) => void
  disabled?: boolean
}

export function ComposerModelPicker({
  models,
  value,
  onChange,
  disabled = false,
}: ComposerModelPickerProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const listId = useId()
  const selection = value === AUTO_MODEL || models.includes(value) ? value : AUTO_MODEL
  const cheapest = pickCheapestModel(models)
  const label = modelLabel(selection)

  useEffect(() => {
    if (!open) return
    const onPointer = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false)
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('mousedown', onPointer)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', onPointer)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  const options: { id: string; title: string; detail: string; icon: 'auto' | 'model' }[] = [
    {
      id: AUTO_MODEL,
      title: 'Auto',
      detail: `Select Auto by LabCD · ${cheapest}`,
      icon: 'auto',
    },
    ...models.map((m) => ({
      id: m,
      title: m,
      detail: 'Fixed model',
      icon: 'model' as const,
    })),
  ]

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        title={selection === AUTO_MODEL ? `Auto → ${cheapest}` : selection}
        className="inline-flex h-8 max-w-[11rem] items-center gap-1 rounded-lg border border-transparent bg-transparent px-2 text-[12.5px] font-medium text-muted-text transition-colors hover:border-border hover:bg-surface-hover hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
        onClick={() => setOpen((v) => !v)}
      >
        {selection === AUTO_MODEL ? (
          <Sparkles className="size-3.5 shrink-0 text-primary" aria-hidden />
        ) : (
          <Zap className="size-3.5 shrink-0 text-muted" aria-hidden />
        )}
        <span className="truncate">{label}</span>
        <ChevronDown
          className={`size-3.5 shrink-0 opacity-70 transition-transform ${open ? 'rotate-180' : ''}`}
          aria-hidden
        />
      </button>

      {open && (
        <div
          id={listId}
          role="listbox"
          aria-label="Select model"
          className="absolute bottom-[calc(100%+6px)] left-0 z-50 min-w-[220px] overflow-hidden rounded-xl border border-border bg-surface-elevated py-1 shadow-[0_12px_40px_rgba(0,0,0,0.45)]"
        >
          {options.map((option) => {
            const active = selection === option.id
            return (
              <button
                key={option.id}
                type="button"
                role="option"
                aria-selected={active}
                className={`flex w-full items-center gap-2.5 px-3 py-2 text-left transition-colors ${
                  active ? 'bg-[color-mix(in_srgb,var(--app-primary)_12%,transparent)]' : 'hover:bg-surface-hover'
                }`}
                onClick={() => {
                  onChange(option.id)
                  setOpen(false)
                }}
              >
                <span className="grid size-7 shrink-0 place-items-center rounded-lg border border-border bg-surface-muted">
                  {option.icon === 'auto' ? (
                    <Sparkles className="size-3.5 text-primary" aria-hidden />
                  ) : (
                    <Zap className="size-3.5 text-muted-text" aria-hidden />
                  )}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px] font-semibold text-foreground">
                    {option.title}
                  </span>
                  <span className="block truncate text-[11px] text-muted">{option.detail}</span>
                </span>
                {active && <Check className="size-3.5 shrink-0 text-primary" aria-hidden />}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
