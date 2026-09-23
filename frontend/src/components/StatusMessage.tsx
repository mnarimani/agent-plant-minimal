import { AlertCircle, CheckCircle2, Info, TriangleAlert } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { statusTypes } from '../lib/classes'

interface StatusMessageProps {
  type: 'info' | 'success' | 'warning' | 'error'
  message: string
}

const STATUS_ICONS: Record<StatusMessageProps['type'], LucideIcon> = {
  info: Info,
  success: CheckCircle2,
  warning: TriangleAlert,
  error: AlertCircle,
}

export function StatusMessage({ type, message }: StatusMessageProps) {
  const Icon = STATUS_ICONS[type]

  return (
    <div
      className={`flex items-start gap-3 px-4 py-3 rounded-lg mb-4 ${statusTypes[type]}`}
      role="status"
    >
      <Icon className="size-5 shrink-0 mt-0.5" aria-hidden />
      <span className="leading-relaxed">{message}</span>
    </div>
  )
}
