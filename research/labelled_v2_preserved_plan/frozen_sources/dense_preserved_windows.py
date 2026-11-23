"""Research-only reversible L2 windows with resumable coarse GA state.

Unlike the first window candidate, this keeps the actual in-progress global
population and absolute target. Local gains retire that target only when it is
superseded. Arm RNGs and main-proposal clocks are separate; extra warm/fresh
evaluations do not become main proposals or trigger immediate contractions.
"""
import hashlib
import math

import torch
import epsilon_ball as base
from research.dense_reversible_windows import ReversibleWindows

LAST_RUN = {}
SOURCE_DEPENDENCIES = ('epsilon_ball.py', 'research/dense_reversible_windows.py')


class PreservedWindows(ReversibleWindows):
    def capture(self, radius, pool, stage, stalled, generation, delta=None):
        self.coarse_capture = dict(query=self.queries, radius=radius, stage=stage,
                                   stalled=stalled, generation=generation, pool_size=len(pool),
                                   pending='contract' if delta is not None else None,
                                   rng_sha256=hashlib.sha256(bytes(self.rng.get_state().tolist())).hexdigest())
        self.coarse_state = dict(radius=radius, pool=list(pool), stage=stage, stalled=stalled,
                                 generation=generation, rng=self.rng, seen_best=self.best_distance,
                                 pending=None, pending_delta=None, main_calls=0,
                                 warm_calls=0, fresh_calls=0, reanchors=0)
        if delta is not None:
            self.contract(self.coarse_state, delta)

    def dense(self):
        # Exact production proposal prefix, including the gate immediately
        # after a successful main proposal but before its pending warm query.
        radius, pool = 30., []
        stage = stalled = generation = successes = 0
        while self.available(reserve=max(2., 14*self.forward_seconds) if self.best is not None else 0.,
                             queries=12 if self.best is not None else 0):
            if successes >= self.switch_successes and generation >= self.minimum_generations:
                self.capture(radius, pool, stage, stalled, generation)
                self.trace.append(dict(event='boundary_start', query=self.queries,
                                       distance=self.best_distance, generation=generation))
                self.boundary(generation)
                return
            if not pool:
                delta = self.initial(radius)
            else:
                parent = pool[0 if self.rand(1).item() < .8 else self.randint(len(pool))][1]
                mate = pool[self.randint(len(pool))][1]
                fraction = max(.0004, .25*.5**(stage//35))
                delta = self.mutate(parent, mate, radius, fraction, generation)
            margin, delta = self.evaluate(delta)
            previous = pool[0][0] if pool else math.inf
            pool.append((margin, delta))
            pool.sort(key=lambda item: item[0])
            pool = pool[:4]
            stalled = 0 if margin < previous-1e-5 else stalled+1
            stage += 1
            generation += 1
            if margin < 0:
                successes += 1
                if successes >= self.switch_successes and generation >= self.minimum_generations:
                    self.capture(radius, pool, stage, stalled, generation, delta)
                    self.trace.append(dict(event='boundary_start', query=self.queries,
                                           distance=self.best_distance))
                    self.boundary(generation)
                    return
                radius *= .8
                pool = []
                if self.available():
                    score, scaled = self.evaluate(self.project(delta*.8, radius))
                    pool = [(score, scaled)]
                stage = stalled = 0
            elif self.best is None and (stage >= 220 or stalled >= 100):
                radius *= 1.6
                pool = []
                stage = stalled = 0
            elif stalled >= 120:
                if self.available():
                    score, fresh = self.evaluate(self.initial(radius))
                    pool = sorted(pool[:2]+[(score, fresh)], key=lambda item: item[0])
                stalled = 0
        self.polish()

    def contract(self, state, delta):
        state['radius'] *= .8
        state.update(pool=[], stage=0, stalled=0, pending='warm',
                     pending_delta=self.project(delta*.8, state['radius']))

    def coarse_step(self, state):
        """Execute exactly one bounded oracle call, preserving original clocks."""
        self.rng = state['rng']
        # Only *external* gains can supersede a pending target. A feasible warm
        # call by this arm must not immediately trigger another contraction.
        external_gain = self.best_distance < state['seen_best']
        if external_gain and self.best_distance <= state['radius']**2:
            old_radius, old_pool = state['radius'], len(state['pool'])
            state.update(radius=.8*math.sqrt(self.best_distance), pool=[], stage=0,
                         stalled=0, pending='warm', reanchors=state['reanchors']+1)
            state['pending_delta'] = self.rescale(self.best-self.x, state['radius'])
            self.state_trace.append(dict(event='coarse_reanchor', query=self.queries,
                                         old_radius=old_radius, radius=state['radius'],
                                         discarded_pool=old_pool, distance=self.best_distance))
        pending = state['pending']
        radius, pool = state['radius'], state['pool']
        if pending == 'warm':
            score, delta = self.evaluate(state['pending_delta'])
            state.update(pool=[(score, delta)], pending=None, pending_delta=None,
                         warm_calls=state['warm_calls']+1)
            kind = 'warm'
        elif pending == 'fresh':
            score, delta = self.evaluate(self.initial(radius))
            state.update(pool=sorted(pool[:2]+[(score, delta)], key=lambda item: item[0]),
                         pending=None, fresh_calls=state['fresh_calls']+1)
            kind = 'fresh'
        else:
            if not pool:
                delta = self.initial(radius)
            else:
                parent = pool[0 if self.rand(1).item() < .8 else self.randint(len(pool))][1]
                mate = pool[self.randint(len(pool))][1]
                fraction = max(.0004, .25*.5**(state['stage']//35))
                delta = self.mutate(parent, mate, radius, fraction, state['generation'])
            margin, delta = self.evaluate(delta)
            previous = pool[0][0] if pool else math.inf
            pool.append((margin, delta))
            pool.sort(key=lambda item: item[0])
            state['pool'] = pool[:4]
            state['stage'] += 1
            state['generation'] += 1
            state['main_calls'] += 1
            state['stalled'] = 0 if margin < previous-1e-5 else state['stalled']+1
            if margin < 0:
                self.contract(state, delta)
            elif state['stalled'] >= 120:
                state.update(pending='fresh', stalled=0)
            kind = 'main'
        state['seen_best'] = self.best_distance
        return kind

    def boundary(self, generation):
        self.state_trace = []
        global_state = self.coarse_state
        # Clone the switch-point stream. The first local window follows the
        # production trajectory until a window limit changes its probe budget.
        local_rng = global_state['rng'].clone_state()
        self.polish()
        self.probe_calls = self.probe_successes = 0
        self.probe_trace = []
        self.windows = []
        self.mate_rescales = self.mate_unused = 0
        alpha = base._setting('BOUNDARY_ALPHA', .25, float)
        fraction = base._setting('BOUNDARY_FRACTION', .99, float)
        if not 0 < alpha <= 1 or not .9 <= fraction < 1:
            raise ValueError('Invalid boundary alpha or fraction')
        local = dict(radius=fraction*math.sqrt(self.best_distance),
                     anchor=self.best_distance, pool=[], feasible=[self.best-self.x],
                     stage=0, stalled=0, alpha=alpha, fraction=fraction, queued=None)
        rates = {'local': None, 'global': None}
        previous, consecutive = None, 0
        local_generation = generation
        while self.work_available():
            operator = self.choose_operator(rates, previous, consecutive)
            before_query, before_distance = self.queries, self.best_distance
            before_main = global_state['generation']
            end_query = before_query+self.WINDOW_CALLS
            while self.work_available(end_query):
                if operator == 'global':
                    self.coarse_step(global_state)
                else:
                    self.rng = local_rng
                    self.local_step(local, local_generation, end_query)
                    local_generation += 1
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
                                     log_gain=gain, rate=rate, smoothed_rate=rates[operator],
                                     global_main_proposals=global_state['generation']-before_main,
                                     global_generation=global_state['generation'],
                                     local_generation=local_generation,
                                     global_radius=global_state['radius'],
                                     global_stage=global_state['stage'],
                                     global_pending=global_state['pending']))
            consecutive = consecutive+1 if previous == operator else 1
            previous = operator
        self.trace.append(dict(event='preserved_end', query=self.queries, windows=len(self.windows)))
        self.polish()


def make_L2_adversarial_from(image_path):
    previous_threads = torch.get_num_threads()
    search = None
    LAST_RUN.clear()
    base.LAST_RUN.clear()
    try:
        torch.set_num_threads(max(1, base._setting('THREADS', 4, int)))
        with torch.inference_mode():
            search = PreservedWindows(image_path, 'l2')
            return search.run()
    finally:
        if search is not None:
            state = getattr(search, 'coarse_state', {})
            LAST_RUN.update(base.LAST_RUN, dense_schedule='state_preserving_windows',
                            window_calls=search.WINDOW_CALLS, windows=getattr(search, 'windows', []),
                            coarse_capture=getattr(search, 'coarse_capture', None),
                            state_trace=getattr(search, 'state_trace', []),
                            coarse_main_calls=state.get('main_calls', 0),
                            coarse_warm_calls=state.get('warm_calls', 0),
                            coarse_fresh_calls=state.get('fresh_calls', 0),
                            coarse_reanchors=state.get('reanchors', 0))
        torch.set_num_threads(previous_threads)
