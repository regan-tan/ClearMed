import { ArrowUp, BookmarkPlus, FileText, History, Layers, Loader2, MessageSquare, SquarePen } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { Citation, Conversation, NotebookMessage } from '../api/notebook'
import { ChatHistoryList } from './ChatHistoryList'
interface Props {
  chats: Conversation[]; activeId: string; sourceCount: number; busy: boolean; error: string | null
  onSend: (text: string) => void; onSave: (message: NotebookMessage) => void; onFlashcard: (message: NotebookMessage) => void
  onCitation: (citation: Citation) => void; onSelectChat: (id: string) => void; onNewChat: () => void; onDeleteChat: (id: string) => void
  onAdd: () => void; savedIds: string[]; flashcardIds: string[]
}

const NO_MESSAGES: NotebookMessage[] = []

export function NotebookChat({ chats, activeId, sourceCount, busy, error, onSend, onSave, onFlashcard, onCitation, onSelectChat, onNewChat, onDeleteChat, onAdd, savedIds, flashcardIds }: Props) {
  const [input, setInput] = useState('')
  const [historyOpen, setHistoryOpen] = useState(false)
  const bottom = useRef<HTMLDivElement>(null)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const active = chats.find(chat => chat.id === activeId)
  const messages = active?.messages ?? NO_MESSAGES
  useEffect(() => { if (messages.length) bottom.current?.scrollIntoView({ block: 'nearest' }) }, [messages, busy])
  // Grow the composer with its content; CSS max-height caps it and scrolls beyond that.
  useEffect(() => { const el = textarea.current; if (!el) return; el.style.height = 'auto'; el.style.height = `${el.scrollHeight}px` }, [input])
  function submit() { if (!input.trim() || busy) return; onSend(input.trim()); setInput('') }
  return <section className="notebook-panel notebook-chat">
    <div className="panel-heading">
      <h2 className="min-w-0"><span className="truncate">{active?.title ?? 'New chat'}</span></h2>
      <div className="flex shrink-0 items-center gap-1">
        {/* On wider screens chat history lives in the sidebar */}
        <button className="icon-button history-toggle" onClick={() => setHistoryOpen(open => !open)} aria-expanded={historyOpen} aria-label="Chat history" title="Chat history"><History size={16} /></button>
        <button className="icon-button" disabled={!active} onClick={() => { onNewChat(); setHistoryOpen(false) }} aria-label="New chat" title="New chat"><SquarePen size={16} /></button>
      </div>
    </div>
    {historyOpen && <div className="chat-history" role="dialog" aria-label="Chat history">
      <p className="chat-history-title">Chat history</p>
      <ChatHistoryList chats={chats} activeId={activeId} onSelect={id => { onSelectChat(id); setHistoryOpen(false) }} onDelete={onDeleteChat} />
    </div>}
    <div className="conversation" onClick={() => setHistoryOpen(false)}>
      {!messages.length
        ? <div className="chat-welcome"><div className="notebook-emblem"><MessageSquare size={28} strokeWidth={1.4} /></div><p className="eyebrow">YOUR CLEARMED NOTEBOOK</p><h1>Make sense of<br />what you’re studying.</h1><p>Bring your sources. Ask a question.<br />Keep the explanation connected to the evidence.</p><div className="starter-grid">{(sourceCount ? ['Summarise the key concepts', 'Explain a difficult concept', 'Compare my sources'] : ['Explain a difficult concept', 'Quiz me on a topic', 'Summarise a condition']).map(text => <button key={text} onClick={() => onSend(text)}>{text}<span>↗</span></button>)}</div>{!sourceCount && <button className="primary-button" onClick={onAdd}>Add your sources <FileText size={15} /></button>}</div>
        : <div className="message-list">{messages.map(message => <article key={message.id} className={`notebook-message ${message.role}`}><p className="message-author">{message.role === 'user' ? 'You' : 'ClearMed'}{message.role === 'assistant' && <span>Interface preview</span>}</p><p className="message-content">{message.content}</p>{message.citations?.length ? <div className="citation-links">{message.citations.map((citation, index) => <button key={`${citation.source_id}-${index}`} onClick={() => onCitation(citation)}><FileText size={13} />{index + 1}. {citation.source_name}</button>)}</div> : null}{message.role === 'assistant' && <div className="message-actions"><button className="save-note" disabled={savedIds.includes(message.id)} onClick={() => onSave(message)}><BookmarkPlus size={14} />{savedIds.includes(message.id) ? 'Saved to notes' : 'Save to notes'}</button><button className="save-note" disabled={flashcardIds.includes(message.id)} onClick={() => onFlashcard(message)}><Layers size={14} />{flashcardIds.includes(message.id) ? 'In flashcards' : 'Make flashcard'}</button></div>}</article>)}{busy && <p className="loading-message" role="status"><Loader2 className="animate-spin" size={16} />Preparing response…</p>}<div ref={bottom} /></div>}
      {error && <p className="chat-error" role="alert">{error}</p>}
    </div>
    <div className="composer-area">
      <form onSubmit={e => { e.preventDefault(); submit() }} className="notebook-composer">
        <textarea ref={textarea} rows={1} value={input} onChange={e => setInput(e.target.value)} aria-label="Ask a question" placeholder={sourceCount ? 'Ask a question about your sources…' : 'Ask a question, or add sources to ground the answer…'} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit() } }} />
        <button type="submit" aria-label="Send question" disabled={!input.trim() || busy}><ArrowUp size={18} /></button>
      </form>
      <p className="composer-caption"><span className={sourceCount ? 'source-count active' : 'source-count'}><FileText size={12} />{sourceCount ? `${sourceCount} source${sourceCount !== 1 ? 's' : ''} selected` : 'No sources selected'}</span><span>Preview responses only · not for clinical decisions</span></p>
    </div>
  </section>
}
