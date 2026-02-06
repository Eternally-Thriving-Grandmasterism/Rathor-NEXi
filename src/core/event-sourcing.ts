// src/core/event-sourcing.ts – Event Sourcing Pattern v1.0
// Append-only event log + state reconstruction + snapshotting
// Valence-aware event validation, mercy-protected replay, lattice-integrated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

interface DomainEvent<T = any> {
  type: string;
  aggregateId: string;
  timestamp: number;
  version: number;
  payload: T;
  valence: number;                    // valence at event creation time
  correlationId?: string;
  causationId?: string;
}

interface Snapshot<T = any> {
  aggregateId: string;
  version: number;
  state: T;
  timestamp: number;
}

interface EventStore {
  append(event: DomainEvent): Promise<void>;
  getEvents(aggregateId: string, fromVersion?: number): Promise<DomainEvent[]>;
  getSnapshot(aggregateId: string): Promise<Snapshot | null>;
  saveSnapshot(snapshot: Snapshot): Promise<void>;
}

class InMemoryEventStore implements EventStore {
  private events: Map<string, DomainEvent[]> = new Map();
  private snapshots: Map<string, Snapshot> = new Map();

  async append(event: DomainEvent): Promise<void> {
    if (!this.events.has(event.aggregateId)) {
      this.events.set(event.aggregateId, []);
    }
    const aggregateEvents = this.events.get(event.aggregateId)!;
    if (event.version !== aggregateEvents.length + 1) {
      throw new Error(`Version conflict for ${event.aggregateId}: expected ${aggregateEvents.length + 1}, got ${event.version}`);
    }
    aggregateEvents.push(event);
  }

  async getEvents(aggregateId: string, fromVersion = 0): Promise<DomainEvent[]> {
    const events = this.events.get(aggregateId) || [];
    return events.slice(fromVersion);
  }

  async getSnapshot(aggregateId: string): Promise<Snapshot | null> {
    return this.snapshots.get(aggregateId) || null;
  }

  async saveSnapshot(snapshot: Snapshot): Promise<void> {
    this.snapshots.set(snapshot.aggregateId, snapshot);
  }
}

const eventStore: EventStore = new InMemoryEventStore(); // replace with real persistent store

interface AggregateRoot<TState> {
  aggregateId: string;
  version: number;
  state: TState;
  uncommittedEvents: DomainEvent[];
  apply(event: DomainEvent): void;
  loadFromHistory(events: DomainEvent[]): void;
}

abstract class AggregateRootBase<TState> implements AggregateRoot<TState> {
  aggregateId: string;
  version = 0;
  state: TState;
  uncommittedEvents: DomainEvent[] = [];

  constructor(aggregateId: string, initialState: TState) {
    this.aggregateId = aggregateId;
    this.state = initialState;
  }

  protected raiseEvent(type: string, payload: any) {
    const valence = currentValence.get();
    if (valence < 0.7) {
      console.warn(`[Aggregate:\( {this.aggregateId}] Low valence ( \){valence.toFixed(3)}) – event ${type} raised cautiously`);
    }

    const event: DomainEvent = {
      type,
      aggregateId: this.aggregateId,
      timestamp: Date.now(),
      version: this.version + 1,
      payload,
      valence
    };

    this.apply(event);
    this.uncommittedEvents.push(event);
  }

  loadFromHistory(events: DomainEvent[]) {
    for (const event of events) {
      this.apply(event);
      this.version = event.version;
    }
  }

  abstract apply(event: DomainEvent): void;

  async commit(): Promise<void> {
    if (!await mercyGate(`Commit aggregate ${this.aggregateId}`)) {
      throw new Error('Mercy gate blocked commit');
    }

    for (const event of this.uncommittedEvents) {
      await eventStore.append(event);
    }

    this.uncommittedEvents = [];
  }
}

// Example aggregate: ValenceAggregate
interface ValenceState {
  currentValence: number;
  history: number[];
  lastUpdated: number;
}

class ValenceAggregate extends AggregateRootBase<ValenceState> {
  constructor(aggregateId: string) {
    super(aggregateId, {
      currentValence: 0.5,
      history: [],
      lastUpdated: Date.now()
    });
  }

  apply(event: DomainEvent) {
    switch (event.type) {
      case 'ValenceUpdated':
        this.state.currentValence = event.payload.newValence;
        this.state.history.push(event.payload.newValence);
        this.state.lastUpdated = event.timestamp;
        break;
      default:
        console.warn(`[ValenceAggregate] Unknown event type: ${event.type}`);
    }
  }

  updateValence(newValence: number) {
    this.raiseEvent('ValenceUpdated', { newValence });
  }
}

/**
 * Load aggregate from event store (with snapshot optimization)
 */
export async function loadAggregate<T extends AggregateRoot<any>>(
  aggregateId: string,
  aggregateClass: new (id: string) => T
): Promise<T> {
  const actionName = `Load aggregate ${aggregateId}`;
  if (!await mercyGate(actionName)) {
    throw new Error('Mercy gate blocked aggregate load');
  }

  const snapshot = await eventStore.getSnapshot(aggregateId);
  let aggregate: T;

  if (snapshot) {
    aggregate = new aggregateClass(aggregateId);
    aggregate.state = snapshot.state;
    aggregate.version = snapshot.version;
    console.log(`[EventSourcing] Loaded snapshot for ${aggregateId} at version ${snapshot.version}`);
  } else {
    aggregate = new aggregateClass(aggregateId);
  }

  const events = await eventStore.getEvents(aggregateId, aggregate.version);
  aggregate.loadFromHistory(events);

  return aggregate;
}

/**
 * Save aggregate changes (events + optional snapshot)
 */
export async function saveAggregate(aggregate: AggregateRoot<any>, createSnapshot = false) {
  await aggregate.commit();

  if (createSnapshot && aggregate.version % 50 === 0) { // snapshot every 50 events
    await eventStore.saveSnapshot({
      aggregateId: aggregate.aggregateId,
      version: aggregate.version,
      state: aggregate.state,
      timestamp: Date.now()
    });
    console.log(`[EventSourcing] Snapshot created for ${aggregate.aggregateId} at version ${aggregate.version}`);
  }
}

// Example usage: valence aggregate saga-like flow
export async function updateAndPersistValence(aggregateId: string, newValence: number) {
  const valenceAgg = await loadAggregate(aggregateId, ValenceAggregate);
  valenceAgg.updateValence(newValence);
  await saveAggregate(valenceAgg, true);
  console.log(`Valence updated to ${newValence} for aggregate ${aggregateId}`);
}
