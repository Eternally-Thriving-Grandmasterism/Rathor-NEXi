// src/sync/electricsql-shape-subscriptions.ts – ElectricSQL Shape-Based Subscriptions v1
// Reactive live queries, shape definitions, offline-first, mercy-gated, valence integration
// MIT License – Autonomicity Games Inc. 2026

import { electricInitializer } from './electricsql-initializer';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import { Observable } from 'rxjs';

const MERCY_THRESHOLD = 0.9999999;

export class ElectricShapeSubscriptions {
  private electric = electricInitializer.getElectricClient();

  /**
   * Subscribe to current user progress shape – live valence/level/experience
   */
  userProgress(userId: string = 'current-user'): Observable<any> {
    return new Observable(observer => {
      if (!this.electric) return observer.error("ElectricSQL not initialized");

      const sub = this.electric.db.users.liveFirst({
        where: `id = '${userId}'`
      }).subscribe(async result => {
        if (result) {
          const passed = await mercyGate('Remote user progress shape update', 'EternalThriving');
          if (passed) {
            await currentValence.setValence(result.valence);
            observer.next(result);
          } else {
            console.warn("[ElectricShape] User progress update blocked by mercy gate");
          }
        }
      });

      return () => sub.unsubscribe();
    });
  }

  /**
   * Subscribe to probes in a specific habitat – live resource/valence changes
   */
  habitatProbes(habitatId: string): Observable<any[]> {
    return new Observable(observer => {
      if (!this.electric) return observer.error("ElectricSQL not initialized");

      const sub = this.electric.db.probes.liveMany({
        where: `habitatId = '${habitatId}'`,
        orderBy: { updatedAt: 'desc' }
      }).subscribe(async results => {
        const passed = await mercyGate('Remote habitat probes shape update', 'EternalThriving');
        if (passed) {
          observer.next(results);
        }
      });

      return () => sub.unsubscribe();
    });
  }

  /**
   * Subscribe to collective valence across habitats (global harmony monitor)
   */
  collectiveHabitatValence(): Observable<number> {
    return new Observable(observer => {
      if (!this.electric) return observer.error("ElectricSQL not initialized");

      const sub = this.electric.db.habitats.liveMany({}).subscribe(async habitats => {
        const collective = habitats.reduce((sum, h) => sum + h.collectiveValence, 0) / habitats.length;

        const passed = await mercyGate('Remote collective valence shape update', 'EternalThriving');
        if (passed) {
          observer.next(collective);
        }
      });

      return () => sub.unsubscribe();
    });
  }
}

export const electricShapeSubs = new ElectricShapeSubscriptions();

// Usage in dashboard / components
/*
electricShapeSubs.userProgress().subscribe(user => {
  console.log('Live user progress:', user);
});

electricShapeSubs.habitatProbes('mars-habitat-7').subscribe(probes => {
  console.log('Live habitat probes:', probes);
});
*/
