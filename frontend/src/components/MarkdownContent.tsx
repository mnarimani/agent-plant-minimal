import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import '../components/landing/landing.css'

type MarkdownContentProps = {
  content: string
  className?: string
}

/**
 * Preprocess math string so that:
 * 1. Inline display math like ': $$formula$$' gets converted into block display math ':\n\n$$\nformula\n$$\n\n'
 * 2. Standalone single-line $$formula$$ gets inner newlines for proper KaTeX display rendering.
 */
function preprocessMath(text: string): string {
  if (!text) return ''
  let normalized = text.replace(/([^\n])\s*\$\$([\s\S]+?)\$\$/g, '$1\n\n$$$$\n$2\n$$$$\n\n')
  normalized = normalized.replace(/\n\$\$([^\n]+?)\$\$/g, '\n$$$$\n$1\n$$$$')
  return normalized
}

export function MarkdownContent({ content, className = 'landing-markdown' }: MarkdownContentProps) {
  const processed = preprocessMath(content)

  return (
    <div className={`${className} math-content`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {processed}
      </ReactMarkdown>
    </div>
  )
}
