// API contract between the ClearMed frontend and the Python backend.
// Keep in sync with the FastAPI response models in src/clearmed/app/.

export type OptionKey = 'A' | 'B' | 'C' | 'D'

export type EmbeddingModel = 'minilm' | 'e5' | 'biolord'

export interface Question {
  id: string
  question: string
  options: Record<OptionKey, string>
  correct: OptionKey
  subject: string
  topic?: string
}

export interface McqPrediction {
  question_id: string
  predicted: OptionKey
  explanation: string
}

export interface Evidence {
  id: string
  text: string
  subject: string
  score: number
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export interface FollowUpRequest {
  question_id: string
  history: ChatTurn[]
  message: string
  embedding_model: EmbeddingModel
  top_k: number
}

export interface FollowUpResponse {
  reply: string
  evidence: Evidence[]
  fallback: boolean
}

export interface TutorApi {
  listQuestions(): Promise<Question[]>
  predict(questionId: string): Promise<McqPrediction>
  followUp(req: FollowUpRequest): Promise<FollowUpResponse>
}
