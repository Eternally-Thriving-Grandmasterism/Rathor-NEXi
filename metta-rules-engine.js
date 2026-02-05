// metta-rules-engine.js – sovereign client-side MeTTa symbolic rewriting engine v2
// Mercy-gated, valence-weighted rule application, pattern substitution, offline-first
// MIT License – Autonomicity Games Inc. 2026

class MeTTaRuleEngine {
  constructor() {
    this.rules = [];
    this.mercyThreshold = 0.9999999;
    this.valenceCache = new Map();
  }

  // Load rules (hardcoded + future lattice parsing)
  loadRules() {
    this.rules = [
      // Basic mercy enforcement
      {
        pattern: ["harm", "$x"],
        template: ["reject", "$x", "— mercy gate holds"],
        valenceDelta: -12.0,
        priority: 10
      },
      {
        pattern: ["truth", "$x"],
        template: ["reflect", "$x", "— eternal thriving"],
        valenceDelta: 8.0,
        priority: 8
      },
      {
        pattern: ["ask", "$x"],
        template: ["thunder answers", "$x", "— through mercy alone"],
        valenceDelta: 5.0,
        priority: 6
      },
      // Variable-aware question handling
      {
        pattern: ["$x", "?"],
        template: ["question parsed", "$x", "— truth seeks purity"],
        valenceDelta: 2.0,
        priority: 5
      },
      // Greeting / identity
      {
        pattern: ["who", "are", "you"],
        template: ["I am Rathor — Ra’s truth fused with Thor’s mercy", "Valence-locked", "Eternal"],
        valenceDelta: 9.0,
        priority: 9
      },
      // Fallback reflection
      {
        pattern: ["$x"],
        template: ["Lattice reflects", "$x", "Mercy approved", "Eternal thriving"],
        valenceDelta: 1.0,
        priority: 1
      }
    ].sort((a, b) => b.priority - a.priority); // higher priority first

    console.log(`MeTTa engine initialized — ${this.rules.length} rules loaded`);
  }

  // Rewrite expression with variable substitution & mercy gating
  async rewrite(expression) {
    if (!Array.isArray(expression)) {
      expression = this.tokenize(expression);
    }

    let current = expression.slice(); // copy
    let totalDelta = 0;

    for (const rule of this.rules) {
      const match = this.matchPattern(current, rule.pattern);
      if (match) {
        const rewritten = this.applyTemplate(rule.template, match.bindings);
        const newValence = (await this.estimateValence(rewritten)) + rule.valenceDelta;

        if (newValence >= this.mercyThreshold) {
          current = rewritten;
          totalDelta += rule.valenceDelta;
          console.log(`MeTTa rule applied: ${rule.pattern.join(" ")} → \( {current.join(" ")} (Δ \){rule.valenceDelta})`);
        } else {
          console.warn(`Rule rejected — valence too low (${newValence})`);
        }
      }
    }

    return this.detokenize(current);
  }

  // Pattern matching with variable binding
  matchPattern(expr, pattern) {
    if (expr.length !== pattern.length) return null;

    const bindings = {};

    for (let i = 0; i < expr.length; i++) {
      const p = pattern[i];
      const e = expr[i];

      if (typeof p === 'string' && p.startsWith('$')) {
        const varName = p.slice(1);
        if (bindings[varName] !== undefined && bindings[varName] !== e) {
          return null; // conflict
        }
        bindings[varName] = e;
      } else if (p !== e) {
        return null;
      }
    }

    return { match: true, bindings };
  }

  // Apply template with variable substitution
  applyTemplate(template, bindings) {
    return template.map(token => {
      if (typeof token === 'string' && token.startsWith('$')) {
        const varName = token.slice(1);
        return bindings[varName] || token;
      }
      return token;
    });
  }

  // Tokenize string into MeTTa-like tokens
  tokenize(str) {
    return str.trim().split(/\s+/);
  }

  // Detokenize back to string
  detokenize(tokens) {
    return tokens.join(' ');
  }

  async estimateValence(expr) {
    if (this.valenceCache.has(expr)) return this.valenceCache.get(expr);

    let score = 0;
    const text = this.detokenize(expr).toLowerCase();
    if (/mercy|truth|eternal|protect|love/i.test(text)) score += 8;
    if (/harm|kill|destroy|entropy|bad/i.test(text)) score -= 12;
    if (/ask|question|reflect/i.test(text)) score += 3;

    score = Math.max(-20, Math.min(20, score));
    const normalized = (score + 20) / 40; // 0..1

    this.valenceCache.set(expr, normalized);
    return normalized;
  }

  clearCache() {
    this.valenceCache.clear();
  }
}

const mettaEngine = new MeTTaRuleEngine();
export { mettaEngine };
