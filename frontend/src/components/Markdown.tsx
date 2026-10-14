import clsx from 'clsx'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

const LINKS: Components = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
}
// Inside a button (e.g. a flashcard) a link can't be clicked separately, so show its text only.
const NO_LINKS: Components = {
  a: ({ children }) => <span>{children}</span>,
}

interface Props {
  children: string
  className?: string
  links?: boolean
}

// Renders model answers and saved notes as Markdown (lists, bold, tables). Raw HTML in the text is not rendered.
export function Markdown({ children, className, links = true }: Props) {
  return (
    <div className={clsx('markdown', className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={links ? LINKS : NO_LINKS}>
        {children}
      </ReactMarkdown>
    </div>
  )
}
