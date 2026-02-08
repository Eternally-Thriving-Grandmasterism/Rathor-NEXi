// metta-hyperon-bridge.js — PATSAGi Council-forged JS bridge for mercy_ethics_core.metta
// Faithful translation with deeply expanded negation detection and polarity inversion
// Pure browser-native, no dependencies — enables soul-nuanced valence gating

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

// Expanded negation cues — words and phrases
const NEGATION_WORDS = [
  'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'noone',
  "don't", "doesn't", "didn't", "isn't", "aren't", "won't", "can't", "shouldn't",
  'hardly', 'barely', 'scarcely', 'lack of', 'no longer', 'without'
];

// Simple string matcher
function matches(context, patterns) {
  const lowerContext = context.toLowerCase();
  return patterns.some(pattern => lowerContext.includes(pattern.toLowerCase()));
}

// Split into rough clauses for scoped negation (sentence-aware)
function getClauses(text) {
  return text.split(/[.!?;:]\s*/).filter(c => c.trim().length > 0);
}

function positiveLongTerm(context) {
  return !matches(context, NEGATIVE_KEYWORDS.slice(0, 10));
}

// Expanded empathyScore with advanced negation handling
function empathyScore(context) {
  const lower = context.toLowerCase();
  let score = 0.5; // Neutral base

  // Base counts
  const positiveCount = POSITIVE_KEYWORDS.filter(k => lower.includes(k)).length;
  const negativeCount = NEGATIVE_KEYWORDS.filter(k => lower.includes(k)).length;
  const empathyCount = EMPATHY_KEYWORDS.filter(k => lower.includes(k)).length;

  // Relational pronoun boost
  if (matches(context, RELATIONAL_PRONOUNS)) {
    score += 0.25;
  }

  // Direct empathy boost
  score += 0.15 * empathyCount;

  // Sentiment polarity base
  const sentimentDiff = positiveCount - negativeCount;
  score += 0.3 * Math.tanh(sentimentDiff / 3);

  // Advanced negation detection with polarity inversion
  let negationAdjustment = 0;
  const clauses = getClauses(lower);

  clauses.forEach(clause => {
    let inversionActive = false;
    const words = clause.split(/\s+/);

    words.forEach((word, idx) => {
      const cleanedWord = word.replace(/[^\w]/g, '');
      
      // Detect negation cue
      if (NEGATION_WORDS.some(neg => cleanedWord.includes(neg) || word.includes(neg))) {
        inversionActive = !inversionActive; // Toggle for chained negations
      }

      // Look forward in scope (next 8 words or clause end)
      const scopeEnd = Math.min(idx + 8, words.length);
      const scope = words.slice(idx + 1, scopeEnd).join(' ');

      if (inversionActive) {
        if (POSITIVE_KEYWORDS.some(k => scope.includes(k)) || EMPATHY_KEYWORDS.some(k => scope.includes(k))) {
          negationAdjustment -= 0.35; // Negated positive/empathy = strong penalty
        }
        if (NEGATIVE_KEYWORDS.some(k => scope.includes(k))) {
          negationAdjustment += 0.30; // Negated negative = boost (not bad = good)
        }
      }
    });
  });

  // Additional global vicinity fallback for short phrases
  NEGATION_WORDS.forEach(neg => {
    const negIndex = lower.indexOf(neg);
    if (negIndex !== -1) {
      const vicinity = lower.substring(negIndex, negIndex + 80); // Forward-focused scope
      if (matches(vicinity, [...POSITIVE_KEYWORDS, ...EMPATHY_KEYWORDS])) {
        negationAdjustment -= 0.25;
      }
      if (matches(vicinity, NEGATIVE_KEYWORDS)) {
        negationAdjustment += 0.20;
      }
    }
  });

  score += negationAdjustment;

  // Final bounding
  score = Math.max(0.0, Math.min(1.0, score));

  console.log(`Empathy score — Pos: ${positiveCount}, Neg: ${negativeCount}, Empathy: ${empathyCount}, NegAdj: ${negationAdjustment.toFixed(3)} → Score: ${score.toFixed(4)}`);

  return score;
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
  console.log('MeTTa-Hyperon JS bridge active — expanded negation thriving. ⚡️');
  return true;
}

initHyperonBridge();
