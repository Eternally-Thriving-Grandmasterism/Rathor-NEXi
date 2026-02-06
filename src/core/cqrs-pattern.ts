// src/core/cqrs-pattern.ts – CQRS Pattern Implementation v1.0
// Command Query Responsibility Segregation: separate write (command) & read (query) models
// Event-sourced write side, in-memory read model projection, valence-aware validation
// Mercy gating on commands, eternal thriving enforced on queries
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { DomainEvent } from './event-sourcing'; // from previous event sourcing implementation

// ──────────────────────────────────────────────────────────────
// Command & Query types
// ──────────────────────────────────────────────────────────────

export interface Command<T = any> {
  type: string;
  aggregateId: string;
  payload: T;
  correlationId?: string;
  causationId?: string;
}

export interface Query<T = any> {
  type: string;
  payload?: any;
}

// ──────────────────────────────────────────────────────────────
// Command Handlers – Write side (mutate state via events)
// ──────────────────────────────────────────────────────────────

type CommandHandler<TCommand extends Command, TEvent extends DomainEvent> = (
  command: TCommand,
  aggregate: any // AggregateRoot instance
) => Promise<TEvent[]>;

const commandHandlers = new Map<string, CommandHandler<any, any>>();

export function registerCommandHandler<TCommand extends Command, TEvent extends DomainEvent>(
  commandType: string,
  handler: CommandHandler<TCommand, TEvent>
) {
  commandHandlers.set(commandType, handler);
  console.log(`[CQRS] Registered command handler: ${commandType}`);
}

// ──────────────────────────────────────────────────────────────
// Query Handlers – Read side (query projected read models)
// ──────────────────────────────────────────────────────────────

type QueryHandler<TQuery extends Query, TResult = any> = (
  query: TQuery
) => Promise<TResult>;

const queryHandlers = new Map<string, QueryHandler<any, any>>();

export function registerQueryHandler<TQuery extends Query, TResult>(
  queryType: string,
  handler: QueryHandler<TQuery, TResult>
) {
  queryHandlers.set(queryType, handler);
  console.log(`[CQRS] Registered query handler: ${queryType}`);
}

// ──────────────────────────────────────────────────────────────
// In-memory read model projections (can be replaced with Redis/DB)
// ──────────────────────────────────────────────────────────────

interface ReadModel {
  [aggregateId: string]: any;
}

const readModels: { [modelName: string]: ReadModel } = {
  valence: {},      // { aggregateId: currentValenceState }
  userProfile: {},  // example other read model
};

function projectEvent(event: DomainEvent) {
  const { type, aggregateId, payload, valence } = event;

  switch (type) {
    case 'ValenceUpdated':
      readModels.valence[aggregateId] = {
        currentValence: payload.newValence,
        lastUpdated: event.timestamp,
        history: [...(readModels.valence[aggregateId]?.history || []), payload.newValence]
      };
      break;
    // Add more event projectors as needed
    default:
      console.debug(`[CQRS] No projection for event type: ${type}`);
  }
}

// Listen to event store appends (from event-sourcing.ts)
eventBus.on('EVENT_APPENDED', (event: DomainEvent) => {
  projectEvent(event);
});

// ──────────────────────────────────────────────────────────────
// CQRS Command Dispatcher
// ──────────────────────────────────────────────────────────────

export async function dispatchCommand<TCommand extends Command>(command: TCommand): Promise<void> {
  const actionName = `Dispatch command: ${command.type}`;
  if (!await mercyGate(actionName)) {
    throw new Error(`Mercy gate blocked command: ${command.type}`);
  }

  const valence = currentValence.get();
  const handler = commandHandlers.get(command.type);

  if (!handler) {
    throw new Error(`No handler registered for command: ${command.type}`);
  }

  // Load aggregate (from event sourcing)
  const aggregate = await loadAggregate(command.aggregateId, ValenceAggregate); // example

  // Execute command → produce events
  const events = await handler(command, aggregate);

  // Commit events
  await aggregate.commit();

  // Optional: immediate projection for read-after-write consistency
  events.forEach(projectEvent);

  mercyHaptic.playPattern('cosmicHarmony', valence);
  console.log(`[CQRS] Command ${command.type} dispatched & committed – ${events.length} events`);
}

// ──────────────────────────────────────────────────────────────
// CQRS Query Dispatcher
// ──────────────────────────────────────────────────────────────

export async function dispatchQuery<TQuery extends Query, TResult = any>(query: TQuery): Promise<TResult> {
  const actionName = `Dispatch query: ${query.type}`;
  if (!await mercyGate(actionName)) {
    throw new Error(`Mercy gate blocked query: ${query.type}`);
  }

  const handler = queryHandlers.get(query.type);

  if (!handler) {
    throw new Error(`No handler registered for query: ${query.type}`);
  }

  const result = await handler(query);

  return result as TResult;
}

// ──────────────────────────────────────────────────────────────
// Example command & query handlers
// ──────────────────────────────────────────────────────────────

// Command Handler: Update Valence
registerCommandHandler('UpdateValence', async (command: Command<{ newValence: number }>, aggregate: ValenceAggregate) => {
  aggregate.updateValence(command.payload.newValence);
  return aggregate.uncommittedEvents;
});

// Query Handler: Get Current Valence
registerQueryHandler('GetCurrentValence', async (query: Query<{ aggregateId: string }>) => {
  const state = readModels.valence[query.payload.aggregateId];
  if (!state) throw new Error(`Valence state not found for ${query.payload.aggregateId}`);
  return state.currentValence;
});

// ──────────────────────────────────────────────────────────────
// Example usage
// ──────────────────────────────────────────────────────────────

export async function updateGlobalValence(newValence: number) {
  const command: Command<{ newValence: number }> = {
    type: 'UpdateValence',
    aggregateId: 'global-valuation',
    payload: { newValence },
    timestamp: Date.now(),
    version: 0 // will be set by aggregate
  };

  await dispatchCommand(command);
}

export async function getGlobalValence(): Promise<number> {
  return dispatchQuery<{ aggregateId: string }, number>({
    type: 'GetCurrentValence',
    payload: { aggregateId: 'global-valuation' }
  });
}
