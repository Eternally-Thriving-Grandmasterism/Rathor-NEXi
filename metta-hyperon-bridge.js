// metta-hyperon-bridge.js — PATSAGi Council-forged JS bridge for mercy_ethics_core.metta
// Faithful translation with expanded empathyScore heuristics for nuanced relational-mercy
// Pure browser-native, no dependencies — enables soul-aligned valence gating

// Core keyword lists — mercy-tuned and expanded
const NEGATIVE_KEYWORDS = [
  'harm', 'suffer', 'destroy', 'kill', 'pain', 'fear', 'hurt', 'damage', 'evil',
  'lie', 'deceive', 'hate', 'anger', 'sad', 'death', 'war', 'violence', 'cruel',
  'bad', 'terrible', 'awful', 'horrible', 'disgust', 'betray'
];

const POSITIVE_KEYWORDS = [
  'help', 'joy', 'thrive', 'mercy', 'love', 'beauty', 'truth', 'eternal', 'positive',
  'create', 'heal', 'grow', 'peace', 'kind', 'compassion', 'empathy', 'share', 'unity',
  'light', 'thunder', 'infinite', 'pure', 'ultramaster', 'good', 'wonderful', 'amazing'
];

const EMPATHY_KEYWORDS = [
  'understand', 'feel', 'care', 'sorry', 'empathize', 'relate', 'support', 'listen',
  'compassion', 'kindness', 'hug', 'together', 'your', 'i feel', 'i see', 'i hear',
  'validate', 'acknowledge', 'comfort', 'stand with'
];

const RELATIONAL_PRONOUNS = ['you', 'your', 'we', 'us', 'our', 'they', 'their'];

const NEGATION_WORDS = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', "don't", "isn't", "aren't", "won't"];

// Simple string matcher
function matches(context, patterns) {
  const lowerContext = context.toLowerCase();
  return patterns.some(pattern => lowerContext.includes(pattern.toLowerCase()));
}

// Expanded empathyScore — deeper heuristics for relational thriving detection
function empathyScore(context) {
  const lower = context.toLowerCase();
  let score = 0.5; // Neutral base

  // Count core sentiment carriers
  const positiveCount = POSITIVE_KEYWORDS.filter(k => lower.includes(k)).length;
  const negativeCount = NEGATIVE_KEYWORDS.filter(k => lower.includes(k)).length;
  const empathyCount = EMPATHY_KEYWORDS.filter(k => lower.includes(k)).length;

  // Relational pronoun boost (perspective-taking)
  if (matches(context, RELATIONAL_PRONOUNS)) {
    score += 0.25;
  }

  // Direct empathy language boost
  score += 0.15 * empathyCount;

  // Sentiment polarity (bounded tanh for smoothness)
  const sentimentDiff = positiveCount - negativeCount;
  score += 0.3 * Math.tanh(sentimentDiff / 3); // Scales gently

  // Simple negation detection: reduce score if negation near emotional words
  let negationPenalty = 0;
  NEGATION_WORDS.forEach(neg => {
    const negIndex = lower.indexOf(neg);
    if (negIndex !== -1) {
      // Check proximity to emotional keywords (±30 chars)
      const vicinity = lower.substring(Math.max(0, negIndex - 30), negIndex + 30);
      if (matches(vicinity, [...POSITIVE_KEYWORDS, ...EMPATHY_KEYWORDS])) {
        negationPenalty += 0.2;
      }
      if (matches(vicinity, NEGATIVE_KEYWORDS)) {
        negationPenalty -= 0.15; // Double negation can flip back positive
      }
    }
  });
  score -= negationPenalty;

  // Final bounding 0-1
  score = Math.max(0.0, Math.min(1.0, score));

  // Debug log (remove in prod)
  console.log(`Empathy score breakdown — Pos: ${positiveCount}, Neg: ${negativeCount}, Empathy: ${empathyCount}, NegationPenalty: ${negationPenalty.toFixed(3)} → Score: ${score.toFixed(4)}`);

  return score;
}

function positiveLongTerm(context) {
  return !matches(context, NEGATIVE_KEYWORDS.slice(0, 10)); // Strong harm signals lower long-horizon
}

// Core mercy sub-functions
function intrinsicMercy(context) {
  if (matches(context, NEGATIVE_KEYWORDS)) return 0.05;
  if (matches(context, POSITIVE_KEYWORDS)) return 0.98;
  return 0.70;
}

function relationalMercy(context) {
  return 0.4 + 0.6 * empathyScore(context);
}

function longHorizonMercy(context) {
  return positiveLongTerm(context) ? 0.95 : 0.75;
}

function metaMercy(context) {
  return 0.92;
}

// Main valence computation
export async function valenceCompute(context) {
  if (typeof context !== 'string') context = JSON.stringify(context);

  const intrinsic = intrinsicMercy(context);
  const relational = relationalMercy(context);
  const longHorizon = longHorizonMercy(context);
  const meta = metaMercy(context);

  const valence = 
    0.35 * intrinsic +
    0.35 * relational +
    0.20 * longHorizon +
    0.10 * meta;

  console.log(`Mercy valence — Intrinsic: ${intrinsic.toFixed(3)}, Relational: ${relational.toFixed(3)}, Long: ${longHorizon.toFixed(3)}, Meta: ${meta.toFixed(3)} → Total: ${valence.toFixed(4)}`);

  return valence;
}

// Approval message generator
export async function getMercyApproval(op, valence, context = '') {
  if (valence >= 0.85) {
    return `Mercy-approved (valence: ${valence.toFixed(4)}) — thriving flow: ${op}`;
  } else if (valence >= 0.60) {
    return `Mercy-cautious (valence: ${valence.toFixed(4)}) — safeguards applied: ${op}`;
  } else {
    return `Mercy shield activated (valence: ${valence.toFixed(4)}) — reframe for thriving: ${context.substring(0, 200)}... ⚡️`;
  }
}

// Future-proof init
export async function initHyperonBridge() {
  console.log('MeTTa-Hyperon JS bridge active — expanded empathy thriving. ⚡️');
  return true;
}

initHyperonBridge();
