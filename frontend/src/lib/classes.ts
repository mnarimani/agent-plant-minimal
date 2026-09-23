export const pageSection = 'space-y-4 sm:space-y-5'
export const pageTitle = 'mt-0 text-xl font-semibold tracking-tight text-foreground sm:text-2xl'
export const pageIntro = 'text-sm text-muted-text leading-relaxed sm:text-base'

export const fieldLabel =
  'flex flex-col gap-1.5 mb-4 last:mb-0 [&>span]:font-medium [&>span]:text-sm [&>span]:text-foreground'
export const fieldInput =
  'w-full px-3 py-2.5 border border-border-input rounded-lg font-inherit bg-surface-elevated text-foreground shadow-sm transition-[border-color,box-shadow] duration-150 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20'
export const fieldCheckbox = 'flex flex-row items-center gap-2 mb-4 text-foreground'

const btnShared =
  'inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 cursor-pointer font-inherit font-medium text-sm transition-all duration-150 disabled:opacity-60 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary whitespace-nowrap'

export const btnBase = `${btnShared} border border-border-input bg-surface-elevated text-foreground shadow-sm hover:border-primary/50 hover:bg-surface-hover hover:shadow-[0_0_16px_rgba(99,102,241,0.18)] active:scale-[0.98]`

export const btnPrimary = `${btnShared} border border-primary/30 bg-gradient-to-r from-primary via-indigo-500 to-[#4a63e0] text-white shadow-[0_2px_14px_rgba(99,102,241,0.35)] hover:shadow-[0_4px_22px_rgba(99,102,241,0.55)] hover:brightness-110 active:scale-[0.98]`

export const btnWide = 'w-full mt-4'
export const btnLink =
  'inline-flex items-center gap-1.5 bg-transparent border-none text-primary px-0 py-2.5 cursor-pointer font-inherit hover:underline'
export const btnCompact = 'px-3 py-1.5 text-[0.82rem]'

export const cardPanel =
  'bg-surface-elevated/90 backdrop-blur-sm border border-border rounded-xl p-3 shadow-sm sm:p-4'
export const cardProcessing =
  'bg-surface-elevated/90 backdrop-blur-sm border border-border rounded-xl p-5 shadow-sm text-center sm:p-8'

export const codePreview =
  'w-full font-mono text-[0.85rem] p-4 border border-border-input rounded-lg bg-code-bg text-code-text resize-y shadow-inner'

export const statusTypes = {
  info: 'bg-[var(--app-status-info-bg)] text-[var(--app-status-info-text)] border border-[color-mix(in_srgb,var(--app-status-info-text)_20%,transparent)]',
  success:
    'bg-[var(--app-status-success-bg)] text-[var(--app-status-success-text)] border border-[color-mix(in_srgb,var(--app-status-success-text)_20%,transparent)]',
  warning:
    'bg-[var(--app-status-warning-bg)] text-[var(--app-status-warning-text)] border border-[color-mix(in_srgb,var(--app-status-warning-text)_20%,transparent)]',
  error:
    'bg-[var(--app-status-error-bg)] text-[var(--app-status-error-text)] border border-[color-mix(in_srgb,var(--app-status-error-text)_20%,transparent)]',
} as const

export const mutedText = 'text-muted'

export const badgeStyles: Record<string, string> = {
  strategy: 'bg-[var(--app-badge-strategy-bg)] text-[var(--app-badge-strategy-text)]',
  explore: 'bg-[var(--app-badge-strategy-bg)] text-[var(--app-badge-strategy-text)]',
  continue: 'bg-[var(--app-badge-continue-bg)] text-[var(--app-badge-continue-text)]',
  terminate: 'bg-[var(--app-badge-terminate-bg)] text-[var(--app-badge-terminate-text)]',
  juror: 'bg-[var(--app-badge-juror-bg)] text-[var(--app-badge-juror-text)]',
  latest: 'bg-[var(--app-badge-latest-bg)] text-[var(--app-badge-latest-text)]',
}

export const codeBlock =
  'mt-2 mb-0 p-2.5 bg-code-bg text-code-text rounded-md text-[0.78rem] overflow-x-auto whitespace-pre-wrap break-words'

export const logPre =
  'mt-2 mb-0 whitespace-pre-wrap break-words font-mono text-[0.82rem] text-foreground-secondary'
