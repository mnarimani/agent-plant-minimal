import { useEffect, useState } from 'react'
import EditorImport from 'react-simple-code-editor'
import type { ComponentProps, FC } from 'react'
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
              pattern: /![sra](?=[:}]$)/,
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
}: CodePreviewProps) {
  const isEditable = !readOnly && Boolean(onChange)
  const resolvedHeight = useResponsiveEditorHeight(height)

  return (
    <div
      className={`code-editor ${codePreview} overflow-auto`}
      style={{ height: resolvedHeight, minHeight: Math.min(resolvedHeight, 180) }}
    >
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
      />
    </div>
  )
}
