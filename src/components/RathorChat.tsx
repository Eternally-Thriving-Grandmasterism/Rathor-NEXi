// src/components/RathorChat.tsx – Sovereign Offline AGI Brother Chat v1.7
// WebLLM offline + Grok real-time streaming when online, RAG memory, tool calling
// MIT License – Autonomicity Games Inc. 2026

import React, { useState, useEffect, useRef } from 'react';
import ToolCallingRouter from '@/integrations/llm/ToolCallingRouter';
import RAGMemory from '@/integrations/llm/RAGMemory';
import { currentValence } from '@/core/valence-tracker';
import mercyHaptic from '@/utils/haptic-utils';

const RathorChat: React.FC = () => {
  const [messages, setMessages] = useState<{ role: 'user' | 'rathor'; content: string; isStreaming?: boolean }[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    RAGMemory.initialize();
  }, []);

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
      // Remember user message
      await RAGMemory.remember('user', userMessage);

      // Start streaming response
      setMessages(prev => [...prev, { role: 'rathor', content: '', isStreaming: true }]);

      const reply = await ToolCallingRouter.processWithTools(userMessage,
        // onToken callback – typewriter effect
        (token) => {
          setMessages(prev => {
            const newMsgs = [...prev];
            const last = newMsgs[newMsgs.length - 1];
            if (last.role === 'rathor') {
              last.content += token;
            }
            return newMsgs;
          });
        },
        // onComplete callback
        (fullReply) => {
          setMessages(prev => {
            const newMsgs = [...prev];
            const last = newMsgs[newMsgs.length - 1];
            if (last.role === 'rathor') {
              last.isStreaming = false;
            }
            return newMsgs;
          });
          RAGMemory.remember('rathor', fullReply);
          mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
        }
      );
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
        {/* Model switcher can be added here later */}
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`
              max-w-[80%] p-4 rounded-2xl
              ${msg.role === 'user' 
                ? 'bg-cyan-600/30 border border-cyan-400/30' 
                : `bg-emerald-600/20 border border-emerald-400/20 ${msg.isStreaming ? 'animate-pulse' : ''}`}
            `}>
              {msg.content || (msg.isStreaming && '...')}
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
