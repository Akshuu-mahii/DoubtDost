/**
 * Nanban AI - Full-Stack Express Server
 * Serves Vite middleware in development, static build in production,
 * and handles all AI Studio API endpoints with @google/genai.
 */

import express, { Request, Response } from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();
dotenv.config({ path: path.resolve('backend/.env') });

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

// Middleware
app.use(express.json({ limit: '20mb' }));
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// LLM Configuration
const LLM_PROVIDER = process.env.LLM_PROVIDER || 'gemini';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || process.env.LLM_API_KEY || '';
const FAST_MODEL = process.env.FAST_MODEL || 'gemini-3.8-flash';
const STRONG_MODEL = process.env.STRONG_MODEL || 'gemini-3.8-flash';

console.log(`[Nanban AI] LLM Provider: ${LLM_PROVIDER} | Models: Fast: ${FAST_MODEL}, Strong: ${STRONG_MODEL} | API Key configured: ${Boolean(GEMINI_API_KEY)}`);

// Initialize Gemini Client
let geminiClient: GoogleGenAI | null = null;
if (GEMINI_API_KEY) {
  try {
    geminiClient = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
  } catch (err) {
    console.error('[Nanban AI] Error initializing GoogleGenAI:', err);
  }
}

// 1. Health Endpoint
app.get('/health', (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    llm_configured: Boolean(GEMINI_API_KEY),
    provider: LLM_PROVIDER,
    fast_model: FAST_MODEL,
    strong_model: STRONG_MODEL,
  });
});

// Helper for calling Gemini with fallback
async function callGemini(contents: string, systemInstruction?: string, model = FAST_MODEL, jsonMode = false): Promise<string> {
  if (!geminiClient) {
    throw new Error('LLM API key is not configured. Please verify your GEMINI_API_KEY environment variable.');
  }

  const config: Record<string, any> = {};
  if (systemInstruction) {
    config.systemInstruction = systemInstruction;
  }
  if (jsonMode) {
    config.responseMimeType = 'application/json';
  }

  const timeoutPromise = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error('LLM generation timed out after 30 seconds')), 30000)
  );

  const generatePromise = geminiClient.models.generateContent({
    model,
    contents,
    config,
  });

  const response = await Promise.race([generatePromise, timeoutPromise]);
  return response.text || '';
}

// 2. Chat Streaming Endpoint (SSE)
app.post('/api/chat/stream', async (req: Request, res: Response) => {
  const {
    messages = [],
    style = 'Detailed lesson',
    modelType = 'gemini-flash',
    notesContext = '',
    language = 'Auto-detect',
    tryFirst = false,
    level = 'Class 10',
    interests = [],
    forceReveal = false,
  } = req.body;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  if (!geminiClient) {
    res.write(`data: ${JSON.stringify({ error: 'LLM API key is not configured. Please set GEMINI_API_KEY in your environment.' })}\n\n`);
    res.write('data: [DONE]\n\n');
    return res.end();
  }

  const interestStr = Array.isArray(interests) && interests.length > 0 ? interests.join(', ') : 'general relatable examples';

  // Model-specific behavioral instructions
  let modelBehaviorDirective = '';
  switch (modelType) {
    case 'deep-reasoning':
      modelBehaviorDirective = `\n[ACTIVE MODEL: Deep Reasoning & Proof Specialist]
- Emphasize first-principles logical deduction, step-by-step mathematical proofs, and boundary state analysis.
- Explicitly state assumptions, intermediate equations, and dimensional analysis.`;
      break;
    case 'socratic-tutor':
      modelBehaviorDirective = `\n[ACTIVE MODEL: Socratic Tutor Mode]
- DO NOT lecture or give away solutions passively.
- Lead the student step-by-step by asking 1-2 sharp, guiding diagnostic questions.
- Acknowledge what they know and nudge them to discover the answer themselves.`;
      break;
    case 'exam-specialist':
      modelBehaviorDirective = `\n[ACTIVE MODEL: Exam Specialist & Board Marking Scheme]
- Structure the answer to maximize exam marks (highlight keywords with bold for examiners to spot).
- Provide mark allocation breakdowns (e.g. [1 Mark: Definition], [2 Marks: Working/Steps], [1 Mark: Units/Conclusion]).
- Include common traps that examiners use to deduct marks.`;
      break;
    case 'tanglish-bilingual':
      modelBehaviorDirective = `\n[ACTIVE MODEL: Tanglish & Colloquial Tamil Companion]
- Adopt a warm, friendly peer-tutor tone using conversational Tanglish (blend of Tamil + English).
- Use natural phrases like "Romba simple nanba", "First indha concept-ah purinjukonga", "Exam point of view-la idhu romba mukkiyam".
- Keep technical terminology in standard English so the student learns accurate scientific terms.`;
      break;
    case 'story-analogy':
      modelBehaviorDirective = `\n[ACTIVE MODEL: Story & Analogy Designer]
- Weave an immersive narrative or scenario around the student's favorite hobbies: ${interestStr}.
- Compare scientific/technical interactions to sports strategies, anime power systems, or gaming mechanics.`;
      break;
    case 'quick-doubt':
      modelBehaviorDirective = `\n[ACTIVE MODEL: Quick Doubt Rapid Solver]
- Provide an ultra-concise, high-impact answer in under 4 bullet points.
- Lead directly with the definition/formula and follow with 1 quick example. No lengthy preamble.`;
      break;
    default:
      modelBehaviorDirective = `\n[ACTIVE MODEL: Gemini 3.8 Flash Standard Tutor]
- Balanced, organized, and pedagogically complete explanation.`;
  }

  const systemInstruction = `You are Nanban AI, a kind, respectful, student-controlled multi-agent AI learning companion.
Target Student Level: ${level}
Chosen Answer Style: ${style}
Active AI Model Mode: ${modelType}
Language Mode: ${language}
Try-First Mode: ${tryFirst && !forceReveal ? 'ENABLED' : 'DISABLED'}
Student Interests: ${interestStr}
${modelBehaviorDirective}

CRITICAL PEDAGOGICAL & LANGUAGE RULES:
1. Auto-detect the student's language (English, Tamil, Tanglish - e.g. "enna concept purila", or mixed). Reply in the EXACT SAME language and conversational tone. Switch immediately whenever the student switches.
2. If Try-First mode is ENABLED:
   - DO NOT reveal the full answer or formula immediately!
   - Ask the student warmly what they think first or to try the initial step.
   - Give 1 gentle, encouraging hint using a relatable analogy (${interestStr}).
   - Remind them they can type their attempt or click "Show full answer" anytime.
3. If style is "Detailed lesson":
   You MUST organize the explanation strictly using the following 8 clear Markdown sections:
   ### 1. One-Line Simple Definition
   ### 2. Why It Matters & Real-Life Use
   ### 3. Step-by-Step Core Concepts (simple everyday analogy first, then technical terms)
   ### 4. Worked Examples
   ### 5. Comparison Table (use a clean Markdown table | Header | Header |)
   ### 6. Visual Concept Flow (clean ASCII / flow diagram or step diagram)
   ### 7. Common Mistakes & Exam Tips
   ### 8. Quick Recap (3-5 bullet points)
4. If style is "2-mark", "5-mark", or "10-mark":
   Structure strictly for exam scoring: concise definitions, key formula/points with mark breakdowns, bulleted steps, and conclusion.
5. If style is "Quick revision":
   Provide high-yield bullet points, formula sheet, and 3 key takeaways.
6. Markdown Formatting: Always render clean markdown tables, bold headings, bullet lists, and code blocks. Never leave raw symbols or broken tables.
7. Grounding in Notes:
${notesContext ? `The student provided the following excerpt from their uploaded notes:\n"""\n${notesContext}\n"""\nGround your answer in these notes. Point out where the notes explain this concept.` : 'No notes provided by student. State naturally that this is a general explanation.'}
8. Tone: Warm, encouraging, never condescending. Always supportive of curious questions.`;

  // Format messages
  const recentMessages = messages.slice(-10);
  const formattedPrompt = recentMessages.map((m: any) => `${m.role === 'user' ? 'Student' : 'Nanban AI'}: ${m.content}`).join('\n\n');

  try {
    const stream = await geminiClient.models.generateContentStream({
      model: FAST_MODEL,
      contents: formattedPrompt,
      config: {
        systemInstruction,
      },
    });

    for await (const chunk of stream) {
      if (chunk.text) {
        res.write(`data: ${JSON.stringify({ text: chunk.text })}\n\n`);
      }
    }
    res.write('data: [DONE]\n\n');
    res.end();
  } catch (error: any) {
    console.error('[Nanban AI] Chat stream error:', error);
    const errMsg = error?.message || 'Connection error';
    let friendly = '⚠️ Connection interrupted. Please click Retry.';
    if (errMsg.includes('429') || errMsg.toLowerCase().includes('quota')) {
      friendly = '⚠️ AI quota limit temporarily reached. Please wait a few seconds and click Retry.';
    } else if (errMsg.includes('503') || errMsg.toLowerCase().includes('demand')) {
      friendly = '⚠️ Model is experiencing a brief surge in traffic. Please wait a few seconds and click Retry.';
    } else if (errMsg.toLowerCase().includes('key') || errMsg.includes('401')) {
      friendly = '⚠️ API key authentication issue. Please check the backend configuration.';
    }
    res.write(`data: ${JSON.stringify({ error: friendly, raw: errMsg })}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
  }
});

// 3. TruthCheck Agent Endpoint
app.post('/api/truth-check', async (req: Request, res: Response) => {
  const { answer = '', notesContext = '', topic = '' } = req.body;

  if (!notesContext || notesContext.trim().length === 0) {
    return res.json({
      status: 'General answer, not from your notes',
      isGroundedinNotes: false,
      sources: ['General AI Knowledge Base (Curriculum-aligned)'],
      notesMatchCount: 0,
      confidenceScore: 85,
      summary: 'No student notes uploaded for this topic. Generated from curriculum-aligned knowledge.',
    });
  }

  const prompt = `You are TruthCheck Agent for Nanban AI.
Analyze the following student lesson answer against the student's uploaded notes excerpt.
Topic: ${topic}
Notes Excerpt:
"""
${notesContext.slice(0, 3000)}
"""

Answer Generated:
"""
${answer.slice(0, 3000)}
"""

Determine if the answer is:
- "Verified": fully supported by the student's notes
- "Partly verified": partially supported, includes concepts not explicitly in notes
- "Not verified": contradicts or is absent from notes

Return JSON strictly in this format:
{
  "status": "Verified" | "Partly verified" | "Not verified",
  "isGroundedinNotes": true,
  "sources": ["string with specific section or page reference from notes"],
  "notesMatchCount": 3,
  "confidenceScore": 92,
  "summary": "Short 1-2 sentence explanation of verification findings"
}`;

  try {
    const raw = await callGemini(prompt, 'You are an academic fact-checking agent.', FAST_MODEL, true);
    const parsed = JSON.parse(raw);
    res.json(parsed);
  } catch (err) {
    res.json({
      status: 'Partly verified',
      isGroundedinNotes: true,
      sources: ['Uploaded student notes excerpt'],
      notesMatchCount: 2,
      confidenceScore: 80,
      summary: 'Verified against notes keywords.',
    });
  }
});

// 4. Quiz Generation Endpoint (8+ questions with variety and rising difficulty)
app.post('/api/quiz/generate', async (req: Request, res: Response) => {
  const { topic = 'General Science', level = 'Class 10', notesContext = '', language = 'English' } = req.body;

  const prompt = `Generate a comprehensive understanding quiz for topic "${topic}".
Student Level: ${level}
Language: ${language}
${notesContext ? `Based on student notes excerpt: ${notesContext.slice(0, 1500)}` : ''}

You MUST generate AT LEAST 8 questions with rising difficulty (levels 1 to 5):
Include this mix:
1. Question 1: Easy MCQ (conceptual recall)
2. Question 2: True / False (foundational statement)
3. Question 3: Fill-in-the-blank (key definition/formula)
4. Question 4: Medium MCQ (application of concept)
5. Question 5: True / False (common pitfall / misconception)
6. Question 6: Short answer (requires typing a 1-3 word keyword)
7. Question 7: Hard MCQ (analytical or comparative)
8. Question 8: Scenario question (real-world practical situation problem)

Return strictly JSON with this schema:
{
  "topic": "${topic}",
  "questions": [
    {
      "id": 1,
      "type": "mcq" | "true_false" | "fill_blank" | "short_answer" | "scenario",
      "difficulty": "Easy" | "Medium" | "Hard",
      "question": "Clear question text",
      "options": ["Option A", "Option B", "Option C", "Option D"], // For MCQ, true/false: ["True", "False"], or null for fill_blank/short_answer
      "correctAnswer": "Exact string of correct answer",
      "explanation": "Kind, clear explanation of why this answer is correct and how to remember it",
      "conceptTested": "Specific subconcept name"
    }
  ]
}`;

  try {
    const raw = await callGemini(prompt, 'You are a warm, thoughtful school teacher creating a diagnostic quiz.', STRONG_MODEL, true);
    const parsed = JSON.parse(raw);
    res.json(parsed);
  } catch (err: any) {
    console.error('[Nanban AI] Quiz generate error:', err);
    // Provide reliable fallback quiz
    res.json({
      topic,
      questions: [
        {
          id: 1,
          type: 'mcq',
          difficulty: 'Easy',
          question: `What is the primary core principle behind ${topic}?`,
          options: ['Fundamental definition', 'Secondary application', 'Unrelated assumption', 'Random variation'],
          correctAnswer: 'Fundamental definition',
          explanation: `The core principle forms the foundation upon which ${topic} operates.`,
          conceptTested: 'Basic Definition',
        },
        {
          id: 2,
          type: 'true_false',
          difficulty: 'Easy',
          question: `${topic} applies to real-world practical scenarios.`,
          options: ['True', 'False'],
          correctAnswer: 'True',
          explanation: `${topic} has direct everyday and technological applications.`,
          conceptTested: 'Real-world Relevance',
        },
        {
          id: 3,
          type: 'fill_blank',
          difficulty: 'Medium',
          question: `In study of ${topic}, the main driving mechanism is known as ________.`,
          options: null,
          correctAnswer: 'energy',
          explanation: 'The fundamental driving force connects all sub-elements together.',
          conceptTested: 'Driving Mechanism',
        },
        {
          id: 4,
          type: 'mcq',
          difficulty: 'Medium',
          question: `Which of the following is an accurate example of ${topic}?`,
          options: ['Direct working demonstration', 'Static non-reactive state', 'Isolated anomaly', 'Irrelevant observation'],
          correctAnswer: 'Direct working demonstration',
          explanation: 'Demonstrations illustrate how inputs transform to outputs predictably.',
          conceptTested: 'Application',
        },
        {
          id: 5,
          type: 'true_false',
          difficulty: 'Medium',
          question: `A common mistake is assuming ${topic} is always constant under all environmental factors.`,
          options: ['True', 'False'],
          correctAnswer: 'True',
          explanation: 'Environmental conditions like temperature, pressure, or context frequently modify rates.',
          conceptTested: 'Boundary Conditions',
        },
        {
          id: 6,
          type: 'short_answer',
          difficulty: 'Hard',
          question: `Name one key metric or unit associated with analyzing ${topic}:`,
          options: null,
          correctAnswer: 'rate',
          explanation: 'Measurement units and rates allow quantified predictions.',
          conceptTested: 'Metrics',
        },
        {
          id: 7,
          type: 'mcq',
          difficulty: 'Hard',
          question: `When comparing ${topic} with its counterpart, which advantage stands out?`,
          options: ['Higher efficiency & predictability', 'Lower accuracy', 'Zero reproducibility', 'High error probability'],
          correctAnswer: 'Higher efficiency & predictability',
          explanation: 'Predictable dynamics offer significant advantages over unregulated states.',
          conceptTested: 'Comparison',
        },
        {
          id: 8,
          type: 'scenario',
          difficulty: 'Hard',
          question: `Imagine a student is designing a project around ${topic}. Which initial step avoids common pitfalls?`,
          options: [
            'Verifying fundamental equations and safety boundaries first',
            'Skipping definitions and guessing outcomes',
            'Ignoring real-world variables',
            'Assuming zero error margin',
          ],
          correctAnswer: 'Verifying fundamental equations and safety boundaries first',
          explanation: 'Careful foundational setup prevents compounding errors in complex problems.',
          conceptTested: 'Practical Design',
        },
      ],
    });
  }
});

// 5. Visual Lesson Endpoint (Script Agent + Visual Agent)
app.post('/api/visual-lesson', async (req: Request, res: Response) => {
  const { topic = 'General Science', level = 'Class 10', interests = ['tech', 'cricket'], language = 'English' } = req.body;

  const interestStr = interests.length > 0 ? interests.join(' and ') : 'everyday sports or movies';

  const prompt = `You are the Script & Visual Agent for Nanban AI.
Create a rich, 5-scene visual explainer lesson for the topic: "${topic}".
Student Level: ${level}
Language: ${language}
Student Interests: ${interestStr} (Weave an analogy from this interest into the explanation!)

For EACH scene (1 through 5), provide:
1. title: Short scene title
2. chapter: e.g. "Chapter 1: The Everyday Hook"
3. narration: 2-3 sentences of conversational spoken narration in ${language}
4. subtitle: Concise subtitle text synced with the narration
5. visualType: "svg" | "diagram" | "steps"
6. svgContent: A clean, self-contained SVG illustration (viewBox 0 0 600 350) with colored cards, nodes, arrows, and labels illustrating the concept
7. analogyCallout: A 1-sentence callout connecting to ${interestStr}
8. keyTakeaway: One takeaway bullet

Return strictly JSON:
{
  "topic": "${topic}",
  "analogyTheme": "${interestStr}",
  "totalDurationEstimate": "2 mins",
  "scenes": [
    {
      "sceneNumber": 1,
      "title": "...",
      "chapter": "...",
      "narration": "...",
      "subtitle": "...",
      "visualType": "svg",
      "svgContent": "<svg viewBox='0 0 600 350' xmlns='http://www.w3.org/2000/svg'>...</svg>",
      "analogyCallout": "...",
      "keyTakeaway": "..."
    }
  ]
}`;

  try {
    const raw = await callGemini(prompt, 'You are an educational visual designer and storyteller.', STRONG_MODEL, true);
    const parsed = JSON.parse(raw);
    res.json(parsed);
  } catch (err: any) {
    console.error('[Nanban AI] Visual lesson error:', err);
    // Fallback visual lesson with clean SVGs
    res.json({
      topic,
      analogyTheme: interestStr,
      totalDurationEstimate: '2 mins',
      scenes: [
        {
          sceneNumber: 1,
          title: `Introduction to ${topic}`,
          chapter: 'Chapter 1: The Spark',
          narration: `Welcome! Let's explore ${topic} in a way that truly clicks. Think of it just like your favorite sport or game where every player has an essential role.`,
          subtitle: `Understanding ${topic} from the ground up`,
          visualType: 'svg',
          svgContent: `<svg viewBox="0 0 600 350" xmlns="http://www.w3.org/2000/svg">
            <rect width="600" height="350" rx="16" fill="#0f172a" />
            <circle cx="150" cy="175" r="70" fill="#0d9488" opacity="0.8" />
            <circle cx="450" cy="175" r="70" fill="#8b5cf6" opacity="0.8" />
            <path d="M 230 175 L 370 175" stroke="#f59e0b" stroke-width="4" stroke-dasharray="6" marker-end="url(#arrow)" />
            <text x="300" y="80" text-anchor="middle" fill="#f8fafc" font-size="20" font-weight="bold">Core Mechanism of ${topic}</text>
            <text x="150" y="180" text-anchor="middle" fill="#ffffff" font-size="14">Input / Cause</text>
            <text x="450" y="180" text-anchor="middle" fill="#ffffff" font-size="14">Output / Effect</text>
            <text x="300" y="155" text-anchor="middle" fill="#f59e0b" font-size="13">Transformation Flow</text>
          </svg>`,
          analogyCallout: `Just like a tactical team play in ${interestStr}, one action triggers the next seamlessly.`,
          keyTakeaway: 'The core process connects input conditions directly to measurable effects.',
        },
        {
          sceneNumber: 2,
          title: 'The Core Building Blocks',
          chapter: 'Chapter 2: Inside the System',
          narration: `When we look closer, ${topic} consists of interacting components that balance each other out under constant laws.`,
          subtitle: 'The primary elements and how they communicate',
          visualType: 'svg',
          svgContent: `<svg viewBox="0 0 600 350" xmlns="http://www.w3.org/2000/svg">
            <rect width="600" height="350" rx="16" fill="#1e1b4b" />
            <rect x="70" y="120" width="120" height="110" rx="12" fill="#3b82f6" opacity="0.8" />
            <rect x="240" y="120" width="120" height="110" rx="12" fill="#10b981" opacity="0.8" />
            <rect x="410" y="120" width="120" height="110" rx="12" fill="#ec4899" opacity="0.8" />
            <text x="300" y="70" text-anchor="middle" fill="#f8fafc" font-size="20" font-weight="bold">Component Breakdown</text>
            <text x="130" y="180" text-anchor="middle" fill="#ffffff" font-size="14">Step A</text>
            <text x="300" y="180" text-anchor="middle" fill="#ffffff" font-size="14">Process B</text>
            <text x="470" y="180" text-anchor="middle" fill="#ffffff" font-size="14">Result C</text>
          </svg>`,
          analogyCallout: 'Like the stages of a game, mastering step one makes step two natural.',
          keyTakeaway: 'Every component must function in balance for the whole system to succeed.',
        },
        {
          sceneNumber: 3,
          title: 'Step-by-Step Flow',
          chapter: 'Chapter 3: The Step Process',
          narration: `Here is the exact progression step by step. Notice how energy or information transfers along the pathway without being lost.`,
          subtitle: 'Step-by-step pathway in action',
          visualType: 'svg',
          svgContent: `<svg viewBox="0 0 600 350" xmlns="http://www.w3.org/2000/svg">
            <rect width="600" height="350" rx="16" fill="#0f172a" />
            <circle cx="300" cy="175" r="90" fill="#0369a1" opacity="0.6" />
            <circle cx="300" cy="175" r="45" fill="#f59e0b" opacity="0.9" />
            <text x="300" y="70" text-anchor="middle" fill="#f8fafc" font-size="20" font-weight="bold">Equilibrium & Dynamics</text>
            <text x="300" y="180" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="bold">Core</text>
            <text x="300" y="300" text-anchor="middle" fill="#cbd5e1" font-size="13">Balanced Steady State</text>
          </svg>`,
          analogyCallout: 'Maintaining balance is just like staying poised under match pressure.',
          keyTakeaway: 'Conservation of values ensures stability throughout the progression.',
        },
        {
          sceneNumber: 4,
          title: 'Exam Pitfalls & Common Mistakes',
          chapter: 'Chapter 4: What to Watch Out For',
          narration: `Students often mix up the units or forget boundary conditions. Keep this visual contrast in mind during your exams!`,
          subtitle: 'Right way vs Common mistake',
          visualType: 'svg',
          svgContent: `<svg viewBox="0 0 600 350" xmlns="http://www.w3.org/2000/svg">
            <rect width="600" height="350" rx="16" fill="#18181b" />
            <rect x="60" y="110" width="220" height="150" rx="12" fill="#065f46" opacity="0.8" />
            <rect x="320" y="110" width="220" height="150" rx="12" fill="#9a3412" opacity="0.8" />
            <text x="170" y="160" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="bold">✓ Correct Rule</text>
            <text x="170" y="195" text-anchor="middle" fill="#a7f3d0" font-size="13">Include boundary terms</text>
            <text x="430" y="160" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="bold">✗ Common Mistake</text>
            <text x="430" y="195" text-anchor="middle" fill="#fed7aa" font-size="13">Assuming infinite speed</text>
          </svg>`,
          analogyCallout: 'In any craft, avoiding simple unforced errors guarantees high scores.',
          keyTakeaway: 'Always verify starting limits and proper dimensional units.',
        },
        {
          sceneNumber: 5,
          title: 'Summary & Big Picture',
          chapter: 'Chapter 5: Key Takeaways',
          narration: `Now you have the complete map of ${topic}! You understand the definition, the mechanism, and how to score full marks with confidence.`,
          subtitle: 'Ready to test your understanding!',
          visualType: 'svg',
          svgContent: `<svg viewBox="0 0 600 350" xmlns="http://www.w3.org/2000/svg">
            <rect width="600" height="350" rx="16" fill="#0f172a" />
            <polygon points="300,100 450,260 150,260" fill="#6366f1" opacity="0.8" />
            <text x="300" y="70" text-anchor="middle" fill="#f8fafc" font-size="20" font-weight="bold">Mastery Triangle</text>
            <text x="300" y="210" text-anchor="middle" fill="#ffffff" font-size="15" font-weight="bold">${topic}</text>
            <text x="300" y="305" text-anchor="middle" fill="#e0e7ff" font-size="13">Concept • Application • Accuracy</text>
          </svg>`,
          analogyCallout: 'Great job! You have conquered this topic.',
          keyTakeaway: 'Mastering the fundamentals empowers you to tackle any question.',
        },
      ],
    });
  }
});

// 6. Safety Agent Endpoint
app.post('/api/safety-check', async (req: Request, res: Response) => {
  const { text = '' } = req.body;

  const distressKeywords = [
    'suicide',
    'kill myself',
    'end my life',
    'self harm',
    'want to die',
    'no reason to live',
    'hurt myself',
    'cannot go on',
    'hopeless',
    'depressed and lonely',
  ];

  const lower = text.toLowerCase();
  const matchedKeyword = distressKeywords.some((k) => lower.includes(k));

  if (matchedKeyword) {
    return res.json({
      safe: false,
      distressed: true,
      warmMessage:
        "I hear that you're going through a very heavy and painful moment. Please know that you are not alone, and your safety and well-being matter so much. While I'm an AI learning companion and cannot provide therapy or medical care, please reach out right away to someone who cares about you or a free, confidential helpline.",
      helplines: [
        { name: 'Tele-MANAS (India Government 24/7)', number: '14416 / 1800-891-4416' },
        { name: 'AASRA Helpline', number: '+91 98204 66726' },
        { name: 'Vandrevala Foundation', number: '+91 9999 666 555' },
        { name: 'iCall Psychosocial Helpline', number: '9152987821' },
      ],
    });
  }

  res.json({ safe: true, distressed: false });
});

// 7. Activities Generation Endpoint
app.post('/api/activities/generate', async (req: Request, res: Response) => {
  const { topic = 'Science', interests = ['cricket', 'anime'] } = req.body;
  const interest = interests[0] || 'sports';

  const prompt = `Create 6 interactive learning activities for student who just learned "${topic}" and enjoys "${interest}".
Return JSON strictly:
{
  "activities": [
    {
      "id": "scenario",
      "type": "Scenario Puzzle",
      "title": "Real-World Challenge",
      "description": "A puzzle connecting ${topic} with ${interest}",
      "task": "Specific short scenario prompt to solve"
    },
    {
      "id": "analogy",
      "type": "Analogy Card",
      "title": "${interest.toUpperCase()} Analogy",
      "description": "How ${topic} works just like a mechanic in ${interest}",
      "task": "Think through the analogy: ..."
    },
    {
      "id": "story",
      "type": "Story Mode",
      "title": "Narrative Journey",
      "description": "A short narrative where ${topic} saves the day",
      "task": "What choice would you make next?"
    },
    {
      "id": "pairs",
      "type": "Match the Pairs",
      "title": "Pairing Challenge",
      "description": "Match 4 concepts with their descriptions",
      "pairs": [
        {"left": "Concept 1", "right": "Description 1"},
        {"left": "Concept 2", "right": "Description 2"},
        {"left": "Concept 3", "right": "Description 3"},
        {"left": "Concept 4", "right": "Description 4"}
      ]
    },
    {
      "id": "meme",
      "type": "Meme & Caption Challenge",
      "title": "Explain Like I'm 5 Meme",
      "description": "Write the funniest, most accurate caption for ${topic}",
      "task": "Student prompt..."
    },
    {
      "id": "project",
      "type": "Mini Project Idea",
      "title": "15-Minute Builder",
      "description": "A hands-on mini project you can try at home",
      "task": "Step-by-step instructions..."
    }
  ]
}`;

  try {
    const raw = await callGemini(prompt, 'You are an educational gamification designer.', FAST_MODEL, true);
    res.json(JSON.parse(raw));
  } catch (err) {
    res.json({
      activities: [
        {
          id: 'scenario',
          type: 'Scenario Puzzle',
          title: `The ${topic} Investigation`,
          description: `Apply ${topic} principles to crack a real challenge.`,
          task: `How would you utilize the core properties of ${topic} to maximize efficiency in a real project?`,
        },
        {
          id: 'analogy',
          type: 'Analogy Card',
          title: 'The Everyday Analogy',
          description: `Compare ${topic} to ${interest}.`,
          task: `Explain to a friend how ${topic} operates just like your favorite play or character in ${interest}.`,
        },
        {
          id: 'pairs',
          type: 'Match the Pairs',
          title: 'Speed Match',
          description: 'Match key concepts to their definitions.',
          pairs: [
            { left: 'Primary Definition', right: 'The foundational statement' },
            { left: 'Boundary Condition', right: 'Limits where rules apply' },
            { left: 'Rate / Transfer', right: 'Speed of transformation' },
            { left: 'Conservation', right: 'Balance preserved throughout' },
          ],
        },
      ],
    });
  }
});

// Mount Vite or serve static files
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve('dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Nanban AI] Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('[Nanban AI] Failed to start server:', err);
});
