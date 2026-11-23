"""Research-only reversible allocation of L2 global and boundary evolution.

The verified production exploration prefix is inherited unchanged. Subsequent
windows credit actual log squared-distance reduction, rather than accept counts.
Every operator gets at least one of three windows; all queries, including warm
starts and radial probes, count against the same window and attack budgets.
"""
import math

import torch
import epsilon_ball as base

LAST_RUN = {}
SOURCE_DEPENDENCIES = ('epsilon_ball.py',)


class ReversibleWindows(base._ExplorationBoundarySearch):
    WINDOW_CALLS = 96
    MAX_CONSECUTIVE = 2

    @staticmethod
    def choose_operator(rates, previous, consecutive):
        for operator in ('local', 'global'):
            if rates[operator] is None:
                return operator
        if consecutive >= ReversibleWindows.MAX_CONSECUTIVE:
            return 'global' if previous == 'local' else 'local'
        # Exact ties favor local refinement, the current verified default.
        return 'global' if rates['global'] > rates['local'] else 'local'

    def work_available(self, end_query=None):
        return ((end_query is None or self.queries < end_query)
                and self.available(reserve=max(2., 14 * self.forward_seconds), queries=12))

    def boundary(self, generation):
        self.polish()  # One shared initialization, never once per window.
        self.probe_calls = self.probe_successes = 0
        self.probe_trace = []
        self.windows = []
        self.mate_rescales = self.mate_unused = 0
        alpha = base._setting('BOUNDARY_ALPHA', .25, float)
        fraction = base._setting('BOUNDARY_FRACTION', .99, float)
        if not 0 < alpha <= 1 or not .9 <= fraction < 1:
            raise ValueError('Invalid boundary alpha or fraction')
        local = dict(radius=0., anchor=math.inf, pool=[], feasible=[self.best-self.x],
                     stage=0, stalled=0, alpha=alpha, fraction=fraction, queued=None)
        global_state = dict(radius=0., anchor=math.inf, pool=[], stage=0,
                            stalled=0, warm=False, fresh=False)
        rates = {'local': None, 'global': None}
        previous, consecutive = None, 0
        while self.work_available():
            operator = self.choose_operator(rates, previous, consecutive)
            before_query, before_distance = self.queries, self.best_distance
            end_query = before_query + self.WINDOW_CALLS
            while self.work_available(end_query):
                if operator == 'global':
                    self.global_step(global_state, generation)
                else:
                    self.local_step(local, generation, end_query)
                generation += 1
            calls = self.queries-before_query
            if not calls:
                break
            gain = math.log(before_distance/self.best_distance) if self.best_distance > 0 else math.inf
            rate = gain/calls
            old_rate = rates[operator]
            rates[operator] = rate if old_rate is None else .5*(old_rate+rate)
            self.windows.append(dict(operator=operator, start_query=before_query,
                                     end_query=self.queries, calls=calls,
                                     before_distance=before_distance, distance=self.best_distance,
                                     log_gain=gain, rate=rate, smoothed_rate=rates[operator]))
            consecutive = consecutive+1 if previous == operator else 1
            previous = operator
        self.trace.append(dict(event='reversible_end', query=self.queries,
                               windows=len(self.windows)))
        self.polish()

    def global_step(self, state, generation):
        # A local improvement changes the global fitness problem. Old margins
        # cannot be reused at a different radius: invalidate and query a warm
        # start explicitly. Otherwise retain failed elites across windows.
        if self.best_distance < state['anchor']:
            state.update(radius=.8*math.sqrt(self.best_distance), anchor=self.best_distance,
                         pool=[], stage=0, stalled=0, warm=True, fresh=False)
        radius, pool = state['radius'], state['pool']
        if state['warm']:
            delta = self.rescale(self.best-self.x, radius)
            state['warm'] = False
        elif state.get('fresh'):
            delta = self.initial(radius)
            pool = pool[:2]
            state['fresh'] = False
        elif not pool:
            delta = self.initial(radius)
        else:
            parent = pool[0 if self.rand(1).item() < .8 else self.randint(len(pool))][1]
            mate = pool[self.randint(len(pool))][1]
            fraction = max(.0004, .25*.5**(state['stage']//35))
            delta = self.mutate(parent, mate, radius, fraction, generation)
        margin, delta = self.evaluate(delta)
        previous = pool[0][0] if pool else math.inf
        pool.append((margin, delta))
        pool.sort(key=lambda item: item[0])
        state['pool'] = pool[:4]
        state['stage'] += 1
        state['stalled'] = 0 if margin < previous-1e-5 else state['stalled']+1
        if margin < 0:
            state.update(radius=.8*math.sqrt(self.best_distance), anchor=self.best_distance,
                         pool=[], stage=0, stalled=0, warm=True, fresh=False)
        elif state['stalled'] >= 120:
            # Charge the fresh proposal to the following step, not an extra
            # unbounded query at the end of a window.
            state.update(fresh=True, stalled=0)

    def local_step(self, state, generation, end_query):
        if self.best_distance < state['anchor']:
            state.update(radius=state['fraction']*math.sqrt(self.best_distance),
                         anchor=self.best_distance, pool=[], stage=0, stalled=0, queued=None)
            state['feasible'] = [self.best-self.x]+state['feasible'][:3]
        pool, feasible, target = state['pool'], state['feasible'], state['radius']
        parent = (pool[0 if self.rand(1).item() < .8 else self.randint(len(pool))][1]
                  if pool else self.rescale(feasible[0], target))
        if pool and generation % 7 == 0:
            mate = pool[self.randint(len(pool))][1]
        else:
            index = self.randint(len(feasible))
            if generation % 7:
                mate = parent
                self.mate_unused += 1
            else:
                mate = self.rescale(feasible[index], target)
                self.mate_rescales += 1
        fraction = max(.0004, .04*.5**(state['stage']//50))
        if state['queued'] is not None:
            child, state['queued'] = state['queued'], None
        else:
            mutant = self.l2_energy_mutation(parent, mate, target, fraction, generation)
            child = self.rescale((1-state['alpha'])*parent+state['alpha']*mutant, target)
        margin, delta = self.evaluate(child)
        previous = pool[0][0] if pool else math.inf
        pool.append((margin, delta))
        pool.sort(key=lambda item: item[0])
        state['pool'] = pool[:4]
        state['stage'] += 1
        state['stalled'] = 0 if margin < previous-1e-5 else state['stalled']+1
        if margin < 0:
            for _ in range(3):
                if not self.work_available(end_query):
                    break
                before = self.best_distance
                probe_margin, _ = self.evaluate((self.best-self.x)*.98)
                self.probe_calls += 1
                self.probe_trace.append(dict(query=self.queries, margin=probe_margin,
                                             before_distance=before, distance=self.best_distance))
                if probe_margin >= 0:
                    break
                self.probe_successes += 1
            state.update(radius=state['fraction']*math.sqrt(self.best_distance),
                         anchor=self.best_distance, pool=[], stage=0, stalled=0, queued=None)
            state['feasible'] = [self.best-self.x]+feasible[:3]
        elif state['stalled'] >= 120:
            state.update(alpha=max(.04, state['alpha']*.7), stalled=0)
        if state['stage'] and state['stage'] % 256 == 0:
            state['fraction'] = (1+state['fraction'])/2
            state['radius'] = math.sqrt(self.best_distance)*state['fraction']
            state['queued'] = self.rescale(state['pool'][0][1], state['radius'])
            state['pool'] = []


def make_L2_adversarial_from(image_path):
    previous_threads = torch.get_num_threads()
    search = None
    LAST_RUN.clear()
    base.LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, base._setting('THREADS', 4, int)))
        with torch.inference_mode():
            search = ReversibleWindows(image_path, 'l2')
            return search.run()
    finally:
        if search is not None:
            LAST_RUN.update(base.LAST_RUN, dense_schedule='reversible_progress_windows',
                            window_calls=search.WINDOW_CALLS,
                            windows=getattr(search, 'windows', []))
        torch.set_num_threads(previous_threads)
