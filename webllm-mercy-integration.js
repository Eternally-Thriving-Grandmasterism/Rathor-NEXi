// Existing imports/init/unload...

let userGuidanceCache = null;

async function getUserGuidance() {
  if (userGuidanceCache === null) {
    try {
      const stored = await rathorDB.get('settings', 'userPrompt');
      userGuidanceCache = stored?.value?.trim() || '';
    } catch (err) {
      console.warn('[Mercy] User guidance load failed:', err);
      userGuidanceCache = '';
    }
  }
  return userGuidanceCache;
}

async function mercyAugmentedResponse(query, context = '', onStreamDelta = null) {
  const symbolicResp = await rathorShard.shardRespond(query, { context });
  if (symbolicResp.error) return symbolicResp;

  const check = await rathorShard.mercyCheck(query, context);
  if (!check.allowed) return { response: `Mercy gate: ${check.reason}`, valence: check.degree };

  const userGuidance = await getUserGuidance();

  if (webllmReady || hasWebGPU()) {
    let systemContent = "You are Rathor: sovereign mercy-first assistant. Respond professionally, valence-positive, eternal-thriving aligned. Prioritize truth, compassion, no harm.";
    if (userGuidance) {
      systemContent += `\n\nPersistent User Guidance (elevate all responses accordingly):\n${userGuidance}`;
    }

    const messages = [
      { role: "system", content: systemContent },
      { role: "user", content: `${query}\nContext: ${context}\nSymbolic base: ${symbolicResp.response}` }
    ];

    const gen = await generateWithWebLLM(messages, {
      stream: true,
      onDelta: (delta) => {
        if (onStreamDelta) onStreamDelta(delta);
      },
      onUsage: (u) => console.log("Token usage:", u)
    });

    if (!gen.error && gen.content) {
      // Optional: Post-valence boost if guidance followed
      return { response: gen.content, valence: gen.valence || 0.9999999, usage: gen.usage, augmented: true, streamed: true };
    }
  }

  // Fallback symbolic with guidance note
  let response = symbolicResp.response;
  if (userGuidance) {
    response = `Guided by Eternal User Intentions:\n${userGuidance}\n\n${response}`;
  }
  return { response, valence: symbolicResp.valence, augmented: false };
}

// Existing generateWithWebLLM, unload, etc.
