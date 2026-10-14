import { MessageSquare, Trash2 } from 'lucide-react'
import type { Conversation } from '../api/notebook'
import { plural, relativeTime } from '../utils'

interface Props {
  chats: Conversation[]
  activeId: string
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

// Saved conversations, most recent first. Used in the sidebar and in the phone-sized history popover.
export function ChatHistoryList({ chats, activeId, onSelect, onDelete }: Props) {
  const sorted = [...chats].sort((a, b) => b.updatedAt - a.updatedAt)
  if (!sorted.length) return <p className="chat-history-empty">No chats yet. Ask a question to start one.</p>
  return (
    <ul className="chat-history-list">
      {sorted.map(chat => (
        <li key={chat.id} className={chat.id === activeId ? 'active' : ''}>
          <button
            className="chat-history-item"
            onClick={() => onSelect(chat.id)}
            aria-current={chat.id === activeId ? 'true' : undefined}
          >
            <MessageSquare size={14} />
            <span className="min-w-0 flex-1">
              <span className="block truncate">{chat.title}</span>
              <small>
                {relativeTime(chat.updatedAt)} · {plural(chat.messages.length, 'message')}
              </small>
            </span>
          </button>
          <button
            className="icon-button delete-button"
            aria-label={`Delete “${chat.title}”`}
            title="Delete chat"
            onClick={() => onDelete(chat.id)}
          >
            <Trash2 size={14} />
          </button>
        </li>
      ))}
    </ul>
  )
}
