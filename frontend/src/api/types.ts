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
  topic?: string | null
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

export interface LlmModel {
  id: string
  label: string
}

export interface FollowUpRequest {
  question_id: string
  history: ChatTurn[]
  message: string
  embedding_model: EmbeddingModel
  top_k: number
  /** Answer/reply model id from `listModels`; omitted means the server's default. */
  llm?: string
}

export interface FollowUpResponse {
  reply: string
  evidence: Evidence[]
  fallback: boolean
}

export interface TutorApi {
  listQuestions(): Promise<Question[]>
  listModels(): Promise<LlmModel[]>
  predict(questionId: string, llm?: string): Promise<McqPrediction>
  followUp(req: FollowUpRequest): Promise<FollowUpResponse>
}
