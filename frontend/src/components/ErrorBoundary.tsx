import { Component, type ErrorInfo, type ReactNode } from 'react'
import { reportFrontendError, shouldReportFrontendErrors } from '../lib/errorTracking'
import { btnBase, btnPrimary, pageIntro, pageTitle } from '../lib/classes'

type Props = {
  children: ReactNode
}

type State = {
  hasError: boolean
  message: string
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: '' }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      message: error.message || 'Something went wrong',
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    if (!shouldReportFrontendErrors()) return
    reportFrontendError({
      message: error.message || 'React render error',
      stack_trace: `${error.stack ?? ''}\n${info.componentStack ?? ''}`,
      page_url: window.location.href,
      extra: { type: 'react_error_boundary' },
    })
  }

  private handleReload = () => {
    window.location.reload()
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children
    }

    return (
      <div className="flex min-h-screen items-center justify-center bg-surface p-6 text-foreground">
        <div className="max-w-md space-y-4 text-center">
          <h1 className={pageTitle}>Something went wrong</h1>
          <p className={pageIntro}>{this.state.message}</p>
          <button type="button" className={`${btnBase} ${btnPrimary}`} onClick={this.handleReload}>
            Reload page
          </button>
        </div>
      </div>
    )
  }
}
