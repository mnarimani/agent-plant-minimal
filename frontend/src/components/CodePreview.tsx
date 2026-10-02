import { useEffect, useMemo, useState } from 'react'
import type { ComponentProps, CSSProperties, FC } from 'react'
import EditorImport from 'react-simple-code-editor'
import { highlight, languages } from 'prismjs'
import { codePreview } from '../lib/classes'

// Ensure Python grammar is registered without bare global UMD scripts that throw ReferenceError
if (!languages.python) {
  languages.python = ({
    comment: {
      pattern: /(^|[^\\])#.*/,
      lookbehind: true,
      greedy: true,
    },
    'string-interpolation': {
      pattern: /(?:f|fr|rf)(?:("""|''')[\s\S]*?\1|("|')(?:\\.|(?!\2)[^\\\r\n])*\2)/i,
      greedy: true,
      inside: {
        interpolation: {
          pattern: /((?:^|[^{])(?:\{\{)*)\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}])+\})+\})+\}/,
          lookbehind: true,
          inside: {
            'format-spec': {
              pattern: /(:)[^:(){}]+(?=\}$)/,
              lookbehind: true,
            },
            'conversion-option': {
              pattern: /![sra](?=[:]$)/,
              alias: 'punctuation',
            },
            rest: null,
          },
        },
        string: /[\s\S]+/,
      },
    },
    'triple-quoted-string': {
      pattern: /(?:[rub]|br|rb)?("""|''')[\s\S]*?\1/i,
      greedy: true,
      alias: 'string',
    },
    string: {
      pattern: /(?:[rub]|br|rb)?("|')(?:\\.|(?!\1)[^\\\r\n])*\1/i,
      greedy: true,
    },
    function: {
      pattern: /((?:^|\s)def[ \t]+)[a-zA-Z_]\w*(?=\s*\()/g,
      lookbehind: true,
    },
    'class-name': {
      pattern: /(\bclass\s+)\w+/i,
      lookbehind: true,
    },
    decorator: {
      pattern: /(^[\t ]*)@\w+(?:\.\w+)*/m,
      lookbehind: true,
      alias: ['annotation', 'punctuation'],
      inside: {
        punctuation: /\./,
      },
    },
    keyword:
      /\b(?:_(?=\s*:)|and|as|assert|async|await|break|case|class|continue|def|del|elif|else|except|exec|finally|for|from|global|if|import|in|is|lambda|match|nonlocal|not|or|pass|print|raise|return|try|while|with|yield)\b/,
    builtin:
      /\b(?:__import__|abs|all|any|apply|ascii|basestring|bin|bool|buffer|bytearray|bytes|callable|chr|classmethod|cmp|coerce|compile|complex|delattr|dict|dir|divmod|enumerate|eval|execfile|file|filter|float|format|frozenset|getattr|globals|hasattr|hash|help|hex|id|input|int|intern|isinstance|issubclass|iter|len|list|locals|long|map|max|memoryview|min|next|object|oct|open|ord|pow|property|range|raw_input|reduce|reload|repr|reversed|round|set|setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|unichr|unicode|vars|xrange|zip)\b/,
    boolean: /\b(?:False|None|True)\b/,
    number:
      /\b0(?:b(?:_?[01])+|o(?:_?[0-7])+|x(?:_?[a-f0-9])+)\b|(?:\b\d+(?:_\d+)*(?:\.(?:\d+(?:_\d+)*)?)?|\B\.\d+(?:_\d+)*)(?:e[+-]?\d+(?:_\d+)*)?j?(?!\w)/i,
    operator: /[-+%=]=?|!=|:=|\*\*?=?|\/\/?=?|<[<=>]?|>[=>]?|[&|^~]/,
    punctuation: /[{}[\];(),.:]/,
  } as unknown as typeof languages.clike)
  languages.py = languages.python
}

type EditorComponent = FC<ComponentProps<typeof EditorImport>>

// CJS package: Vite may leave the real component nested under `.default`.
const Editor = (
  (EditorImport as unknown as { default?: EditorComponent }).default ?? EditorImport
) as EditorComponent

interface CodePreviewProps {
  value: string
  onChange?: (value: string) => void
  readOnly?: boolean
  height?: number
  language?: 'python' | 'matlab'
  /** Show a narrow gutter of 1-based line numbers (default true). */
  showLineNumbers?: boolean
  className?: string
}

function highlightCode(code: string, language: 'python' | 'matlab') {
  const grammar = language === 'matlab' ? languages.clike : languages.python
  const lang = language === 'matlab' ? 'clike' : 'python'
  return highlight(code, grammar, lang)
}

function useResponsiveEditorHeight(height: number) {
  const [resolved, setResolved] = useState(height)

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 640px)')
    const update = () => {
      setResolved(mq.matches ? Math.min(height, 260) : height)
    }
    update()
    mq.addEventListener('change', update)
    return () => mq.removeEventListener('change', update)
  }, [height])

  return resolved
}

export function CodePreview({
  value,
  onChange,
  readOnly = false,
  height = 400,
  language = 'python',
  showLineNumbers = true,
  className = '',
}: CodePreviewProps) {
  const isEditable = !readOnly && Boolean(onChange)
  const resolvedHeight = useResponsiveEditorHeight(height)

  const lineNumbers = useMemo(() => {
    const n = value ? value.split('\n').length : 1
    return Array.from({ length: Math.max(n, 1) }, (_, i) => i + 1)
  }, [value])

  const gutterWidthCh = Math.max(2, String(lineNumbers.length).length)

  const editorStyle: CSSProperties = {
    padding: '12px 14px',
    minHeight: '100%',
  }

  return (
    <div
      className={`code-editor ${showLineNumbers ? 'code-editor--lined' : ''} ${codePreview} ${className}`.trim()}
      style={{ height: resolvedHeight, minHeight: Math.min(resolvedHeight, 180) }}
    >
      {showLineNumbers && (
        <div
          className="code-editor__gutter"
          aria-hidden="true"
          style={{ ['--gutter-ch' as string]: gutterWidthCh } as CSSProperties}
        >
          {lineNumbers.map((n) => (
            <span key={n} className="code-editor__lineno">
              {n}
            </span>
          ))}
        </div>
      )}
      <div className="code-editor__body overflow-auto">
        <Editor
          value={value}
          onValueChange={(next) => onChange?.(next)}
          highlight={(code) => highlightCode(code, language)}
          disabled={!isEditable}
          padding={0}
          tabSize={4}
          insertSpaces
          className="code-editor__surface"
          textareaClassName="code-editor__textarea"
          preClassName="code-editor__pre"
          style={editorStyle}
        />
      </div>
    </div>
  )
}
