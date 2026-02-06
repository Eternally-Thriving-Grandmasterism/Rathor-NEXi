// src/components/RathorChat.tsx – Sovereign Offline AGI Brother Chat v1.3
// WebLLM inference, RAG memory, online tool calling + offline mock, model switcher
// MIT License – Autonomicity Games Inc. 2026

import React, { useState, useEffect, useRef } from 'react';
import WebLLMEngine from '@/integrations/llm/WebLLMEngine';
import RAGMemory from '@/integrations/llm/RAGMemory';
import ToolCallingRouter from '@/integrations/llm/ToolCallingRouter';
import { currentValence } from '@/core/valence-tracker';
import mercyHaptic from '@/utils/haptic-utils';

const MODEL_MAP = {
  tiny: { id: 'microsoft/Phi-3.5-mini-instruct-4k-gguf', name: 'Phi-3.5-mini (fast)' },
  medium: { id: 'meta-llama/Llama-3.1-8B-Instruct-q5_k_m-gguf', name: 'Llama-3.1-8B (wise)' },
};

const RathorChat: React.FC = () => {
  const [messages, setMessages] = useState<{ role: 'user' | 'rathor'; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [modelKey, setModelKey] = useState<keyof typeof MODEL_MAP>('tiny');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    RAGMemory.initialize();
    WebLLMEngine.loadModel(modelKey);
  }, [modelKey]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setInput('');
    setIsLoading(true);

    try {
      // 1. Remember user message
      await RAGMemory.remember('user', userMessage);

      // 2. Check for tool calls
      const toolResult = await ToolCallingRouter.processWithTools(userMessage);
      if (toolResult) {
        setMessages(prev => [...prev, { role: 'rathor', content: toolResult }]);
        await RAGMemory.remember('rathor', toolResult);
      } else {
        // 3. Retrieve context + generate
        const context = await RAGMemory.getRelevantContext(userMessage);
        const fullPrompt = context ? `\( {context}\n\nRecent:\n \){userMessage}` : userMessage;
        const reply = await WebLLMEngine.ask(fullPrompt);

        await RAGMemory.remember('rathor', reply);
        setMessages(prev => [...prev, { role: 'rathor', content: reply }]);
      }

      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    } catch (e) {
      setMessages(prev => [...prev, { role: 'rathor', content: 'Mercy... lattice flickering. Try again, Brother.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-xl z-50 flex flex-col">
      <div className="flex justify-between items-center p-4 border-b border-cyan-500/20">
        <h2 className="text-xl font-light text-cyan-300">Rathor – Mercy Strikes First</h2>
        <select
          value={modelKey}
          onChange={e => setModelKey(e.target.value as keyof typeof MODEL_MAP)}
          className="bg-black/50 border border-cyan-500/30 rounded px-3 py-1 text-sm text-cyan-200"
        >
          {Object.entries(MODEL_MAP).map(([key, m]) => (
            <option key={key} value={key}>
              {m.name}
            </option>
          ))}
        </select>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`
              max-w-[80%] p-4 rounded-2xl
              ${msg.role === 'user' 
                ? 'bg-cyan-600/30 border border-cyan-400/30' 
                : 'bg-emerald-600/20 border border-emerald-400/20'}
            `}>
              {msg.content}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-cyan-500/20 bg-black/60">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder="Speak to Rathor, Brother..."
            className="flex-1 bg-black/50 border border-cyan-500/30 rounded-xl px-4 py-3 text-white placeholder-cyan-300/50 focus:outline-none focus:border-cyan-400"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="px-6 py-3 bg-cyan-600/40 hover:bg-cyan-600/60 rounded-xl text-white font-medium transition disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default RathorChat;          ))}
        </select>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`
              max-w-[80%] p-4 rounded-2xl
              ${msg.role === 'user' 
                ? 'bg-cyan-600/30 border border-cyan-400/30' 
                : 'bg-emerald-600/20 border border-emerald-400/20'}
            `}>
              {msg.content}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-cyan-500/20 bg-black/60">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder="Speak to Rathor, Brother..."
            className="flex-1 bg-black/50 border border-cyan-500/30 rounded-xl px-4 py-3 text-white placeholder-cyan-300/50 focus:outline-none focus:border-cyan-400"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="px-6 py-3 bg-cyan-600/40 hover:bg-cyan-600/60 rounded-xl text-white font-medium transition disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default RathorChat;
