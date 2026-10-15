// Mock backend used until the real FastAPI service exists.
// Questions are hand-written samples in MedMCQA style, not real dataset rows.

import type { Evidence, FollowUpRequest, FollowUpResponse, McqPrediction, Question, TutorApi } from './types'

const QUESTIONS: Question[] = [
  {
    id: 'q1',
    question: 'Which nerve is most commonly injured in a fracture of the surgical neck of the humerus?',
    options: { A: 'Radial nerve', B: 'Axillary nerve', C: 'Ulnar nerve', D: 'Musculocutaneous nerve' },
    correct: 'B',
    subject: 'Anatomy',
    topic: 'Upper limb',
  },
  {
    id: 'q2',
    question: 'The drug of choice for absence seizures in children is:',
    options: { A: 'Phenytoin', B: 'Carbamazepine', C: 'Ethosuximide', D: 'Phenobarbitone' },
    correct: 'C',
    subject: 'Pharmacology',
    topic: 'Antiepileptics',
  },
  {
    id: 'q3',
    question: 'Wernicke encephalopathy is caused by deficiency of:',
    options: { A: 'Thiamine', B: 'Pyridoxine', C: 'Cobalamin', D: 'Niacin' },
    correct: 'A',
    subject: 'Biochemistry',
    topic: 'Vitamins',
  },
  {
    id: 'q4',
    question: 'The Philadelphia chromosome, t(9;22), is characteristic of:',
    options: {
      A: 'Acute myeloid leukaemia',
      B: 'Chronic lymphocytic leukaemia',
      C: 'Chronic myeloid leukaemia',
      D: 'Hodgkin lymphoma',
    },
    correct: 'C',
    subject: 'Pathology',
    topic: 'Haematological malignancy',
  },
]

const PREDICTIONS: Record<string, McqPrediction> = {
  q1: {
    question_id: 'q1',
    predicted: 'B',
    explanation:
      'The axillary nerve winds around the surgical neck of the humerus with the posterior circumflex humeral artery, so it is vulnerable in fractures at this site.',
  },
  q2: {
    question_id: 'q2',
    predicted: 'C',
    explanation:
      'Ethosuximide blocks T-type calcium channels in thalamic neurons and is first-line for typical absence seizures.',
  },
  q3: {
    question_id: 'q3',
    predicted: 'A',
    explanation:
      'Thiamine (vitamin B1) deficiency, often in chronic alcohol use, causes Wernicke encephalopathy: confusion, ophthalmoplegia and ataxia.',
  },
  q4: {
    question_id: 'q4',
    predicted: 'C',
    explanation:
      'The t(9;22) translocation creates the BCR-ABL fusion gene, which is the hallmark of chronic myeloid leukaemia.',
  },
}

interface CorpusPassage extends Omit<Evidence, 'score'> {
  keywords: string[]
}

const CORPUS: CorpusPassage[] = [
  {
    id: 'train-10231',
    subject: 'Anatomy',
    text: 'Axillary nerve injury leads to paralysis of the deltoid and teres minor, with loss of sensation over the regimental badge area.',
    keywords: ['axillary', 'deltoid', 'teres', 'sensation', 'badge', 'shoulder', 'injury'],
  },
  {
    id: 'train-10544',
    subject: 'Anatomy',
    text: 'Radial nerve is commonly injured in mid-shaft fractures of the humerus, causing wrist drop.',
    keywords: ['radial', 'shaft', 'wrist', 'drop', 'humerus', 'fracture'],
  },
  {
    id: 'train-22817',
    subject: 'Pharmacology',
    text: 'Ethosuximide acts by inhibiting T-type calcium currents in thalamic neurons. Valproate is an alternative when absence seizures coexist with tonic-clonic seizures.',
    keywords: ['ethosuximide', 'calcium', 'thalamic', 'valproate', 'absence', 'mechanism', 'alternative'],
  },
  {
    id: 'train-30112',
    subject: 'Biochemistry',
    text: 'Thiamine pyrophosphate is a cofactor for pyruvate dehydrogenase, alpha-ketoglutarate dehydrogenase and transketolase.',
    keywords: ['thiamine', 'cofactor', 'pyruvate', 'transketolase', 'enzyme', 'mechanism'],
  },
  {
    id: 'train-30480',
    subject: 'Medicine',
    text: 'Glucose should not be given before thiamine in suspected Wernicke encephalopathy, as it may precipitate or worsen the condition.',
    keywords: ['glucose', 'thiamine', 'wernicke', 'treatment', 'before', 'give'],
  },
  {
    id: 'train-41907',
    subject: 'Pathology',
    text: 'Chronic myeloid leukaemia is characterised by the BCR-ABL fusion gene from t(9;22); imatinib inhibits the BCR-ABL tyrosine kinase.',
    keywords: ['cml', 'chronic', 'myeloid', 'bcr-abl', 'imatinib', 'translocation', 'treatment', 'philadelphia'],
  },
]

// Each model ranks the same corpus; the multiplier only makes the comparison visible in the UI.
const MODEL_SCALE: Record<FollowUpRequest['embedding_model'], number> = {
  minilm: 0.82,
  e5: 0.9,
  biolord: 0.96,
}

const FALLBACK_THRESHOLD = 0.35

export const FALLBACK_MESSAGE =
  "I couldn't find enough supporting evidence to answer that reliably. Try rephrasing your question, or check a standard textbook for this topic."

const delay = (ms: number) => new Promise(r => setTimeout(r, ms))

function retrieve(
  query: string,
  subject: string | undefined,
  model: FollowUpRequest['embedding_model'],
  k: number,
): Evidence[] {
  const tokens = new Set(query.toLowerCase().match(/[a-z0-9-]+/g) ?? [])
  return CORPUS.map(({ keywords, ...p }) => {
    const hits = keywords.filter(kw => tokens.has(kw)).length
    const subjectBonus = p.subject === subject ? 0.05 : 0
    const score = Math.min(0.99, (0.1 + hits * 0.22 + subjectBonus) * MODEL_SCALE[model])
    return { ...p, score: Number(score.toFixed(3)) }
  })
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
}

export const mockApi: TutorApi = {
  async listQuestions() {
    await delay(150)
    return QUESTIONS
  },

  async predict(questionId) {
    await delay(500)
    return PREDICTIONS[questionId]
  },

  async followUp(req): Promise<FollowUpResponse> {
    await delay(900)
    const question = QUESTIONS.find(q => q.id === req.question_id)
    const evidence = retrieve(req.message, question?.subject, req.embedding_model, req.top_k)
    const best = evidence[0]?.score ?? 0

    if (best < FALLBACK_THRESHOLD) {
      return { reply: FALLBACK_MESSAGE, evidence, fallback: true }
    }
    return {
      reply: `Based on the retrieved evidence: ${evidence[0].text}`,
      evidence,
      fallback: false,
    }
  },
}
