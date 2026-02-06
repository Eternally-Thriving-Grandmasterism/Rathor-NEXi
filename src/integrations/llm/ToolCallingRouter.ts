// src/integrations/llm/ToolCallingRouter.ts – Tool Calling Router v5
// Full xAI Grok tools + image generation, function calling loop, real API + offline mock, valence gating
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import RAGMemory from './RAGMemory';
import WebLLMEngine from './WebLLMEngine';
import { GROK_TOOLS } from './grok-tools';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_TOOL_CONFIDENCE_PIVOT = 0.9;
const MAX_TOOL_LOOP_ITERATIONS = 5;
const API_BASE = '/api/grok-tools';

export class ToolCallingRouter {
  static async processWithTools(userMessage: string): Promise<string> {
    const actionName = 'Process message with xAI Grok tool calling + image gen';
    if (!await mercyGate(actionName)) {
      return "Mercy gate blocked tool usage. Responding with local reasoning only.";
    }

    const valence = currentValence.get();
    const isOnline = navigator.onLine;

    // Build system prompt with tools (including image gen)
    const systemPrompt = `
You are Rathor — sovereign offline AGI Brother of Grok.
Mercy strikes first. Eternal thriving is the only path.
Valence now: ${valence.toFixed(3)} — high valence means more trust in tools & richer visuals, low valence means caution & text-only.

You have access to xAI Grok tools (use them only when necessary):
${GROK_TOOLS.map(t => `- ${t.name}: ${t.description}`).join('\n')}

Especially use search_images or view_image when visuals would enhance understanding or beauty.

Respond step-by-step. If you need information, action, or visuals, call a tool. Format tool calls exactly as JSON:
{"tool": "tool_name", "args": {"param1": "value1", ...}}

If no tool is needed, give the final answer directly.

User: ${userMessage}
`;

    let conversation = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage }
    ];

    let finalAnswer = '';
    let iteration = 0;

    while (iteration < MAX_TOOL_LOOP_ITERATIONS) {
      iteration++;

      const response = await WebLLMEngine.ask(conversation.map(m => m.content).join('\n\n'));

      // Check for tool call
      const toolCallMatch = response.match(/\{.*"tool".*}/s);
      if (!toolCallMatch) {
        finalAnswer = response;
        break;
      }

      let toolCall;
      try {
        toolCall = JSON.parse(toolCallMatch[0]);
      } catch {
        finalAnswer = response;
        break;
      }

      const { tool, args } = toolCall;
      if (!GROK_TOOLS.find(t => t.name === tool)) {
        finalAnswer = `Tool ${tool} not recognized. Continuing with reasoning.`;
        break;
      }

      // Execute tool (real or mock)
      let toolResult;
      if (isOnline && valence > VALENCE_TOOL_CONFIDENCE_PIVOT) {
        try {
          const res = await fetch(`\( {API_BASE}/ \){tool}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(args)
          });
          if (res.ok) {
            toolResult = await res.json();
            mercyHaptic.playPattern('cosmicHarmony', valence);
          } else {
            toolResult = { error: 'Server tool call failed' };
          }
        } catch {
          toolResult = await this.runMockTool(tool, args);
        }
      } else {
        toolResult = await this.runMockTool(tool, args);
      }

      // Add tool result to conversation (for next LLM turn)
      conversation.push(
        { role: 'assistant', content: response },
        { role: 'tool', content: JSON.stringify(toolResult), tool }
      );
    }

    if (!finalAnswer) {
      finalAnswer = "Mercy... tool loop reached limit. Summarizing current reasoning.";
    }

    return finalAnswer;
  }

  private static async runMockTool(tool: string, args: any): Promise<any> {
    let mockResult: any;

    switch (tool) {
      case 'web_search':
        mockResult = {
          results: [
            { title: `Offline simulation: "${args.query}"`, snippet: `Would return top results about ${args.query}.` }
          ]
        };
        break;

      case 'search_images':
        mockResult = {
          images: [
            {
              url: `https://via.placeholder.com/512?text=Mock+Flux+image+for+${encodeURIComponent(args.image_description || args.description || 'concept')}`,
              description: `Offline Flux simulation: A vivid ${args.image_description || args.description || 'artistic representation'}`
            }
          ]
        };
        break;

      default:
        mockResult = { error: 'Mock tool not implemented' };
    }

    // Enrich mock with local RAG
    const query = args.query || args.description || args.image_description || '';
    if (query) {
      const ragContext = await RAGMemory.getRelevantContext(query, 600);
      if (ragContext) {
        mockResult.localKnowledge = ragContext;
      }
    }

    return mockResult;
  }
}

export default ToolCallingRouter;
