// src/sync/electricsql-real-time-subscriptions.ts – ElectricSQL Real-Time Subscriptions v1
// Live shape-based queries, reactive updates, offline queueing, mercy-gated, valence + Yjs integration
// MIT License – Autonomicity Games Inc. 2026

import { electricInitializer } from './electricsql-initializer';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import { multiplanetarySync } from '@/core/multiplanetary-sync-engine';
import { hybridBridge } from '@/sync/hybrid-yjs-automerge-bridge';
import { Observable } from 'rxjs';

const MERCY_THRESHOLD = 0.9999999;

export class ElectricSQLRealTimeSubscriptions {
  private electric = electricInitializer.getElectricClient();

  /**
   * Subscribe to current user progress (valence, level, experience) – live updates
   */
  subscribeToUserProgress(userId: string = 'current-user'): Observable<any> {
    return new Observable(observer => {
      if (!this.electric) {
        observer.error("ElectricSQL not initialized");
        return;
      }

      const subscription = this.electric.db.users.liveFirst({
        where: { id: userId }
      }).subscribe(result => {
        if (result) {
          // Mercy gate check on incoming remote change
          mercyGate('Remote user progress update', 'EternalThriving').then(passed => {
            if (passed) {
              currentValence.setValence(result.valence);
              observer.next(result);
            } else {
              console.warn("[ElectricSub] Remote progress update blocked by mercy gate");
            }
          });
        }
      });

      return () => subscription.unsubscribe();
    });
  }

  /**
   * Subscribe to all probes in a habitat – live resource & valence changes
   */
  subscribeToHabitatProbes(habitatId: string): Observable<any[]> {
    return new Observable(observer => {
      if (!this.electric) {
        observer.error("ElectricSQL not initialized");
        return;
      }

      const subscription = this.electric.db.probes.liveMany({
        where: { habitatId }
      }).subscribe(results => {
        mercyGate('Remote habitat probes update', 'EternalThriving').then(passed => {
          if (passed) {
            // Propagate to Yjs real-time UI layer
            results.forEach(probe => {
              hybridBridge.syncAutomergeToYjs(`probe-${probe.id}`);
            });
            observer.next(results);
          }
        });
      });

      return () => subscription.unsubscribe();
    });
  }

  /**
   * Subscribe to collective valence spikes (dashboard global harmony monitor)
   */
  subscribeToCollectiveValence(): Observable<number> {
    return new Observable(observer => {
      if (!this.electric) {
        observer.error("ElectricSQL not initialized");
        return;
      }

      // Aggregate live query (example – real impl would use materialized view or server-side aggregation)
      const subscription = this.electric.db.habitats.liveMany({}).subscribe(habitats => {
        const collectiveValence = habitats.reduce((sum, h) => sum + h.collectiveValence, 0) / habitats.length;

        mercyGate('Remote collective valence update', 'EternalThriving').then(passed => {
          if (passed) {
            multiplanetarySync.syncState({ collectiveValence });
            observer.next(collectiveValence);
          }
        });
      });

      return () => subscription.unsubscribe();
    });
  }

  /**
   * Subscribe to progress logs (audit trail / daily pulse history)
   */
  subscribeToProgressLogs(userId: string = 'current-user'): Observable<any[]> {
    return new Observable(observer => {
      if (!this.electric) return observer.error("ElectricSQL not initialized");

      const subscription = this.electric.db.progress_logs.liveMany({
        where: { userId },
        orderBy: { timestamp: 'desc' },
        limit: 50
      }).subscribe(logs => {
        mercyGate('Remote progress logs update', 'EternalThriving').then(passed => {
          if (passed) observer.next(logs);
        });
      });

      return () => subscription.unsubscribe();
    });
  }
}

export const electricRealTimeSubs = new ElectricSQLRealTimeSubscriptions();

// Usage examples in dashboard / components
/*
electricRealTimeSubs.subscribeToUserProgress().subscribe(user => {
  console.log('Live user progress:', user);
});

electricRealTimeSubs.subscribeToHabitatProbes('mars-habitat-7').subscribe(probes => {
  console.log('Live habitat probes:', probes);
});
*/
