import numpy as np
import uuid


class SattvaEngine:
    """
    General-purpose resonance engine. Discovers structure from raw vectors.

    Architecture:
        - Primitives: crystallized base patterns, heavily myelinated, shared across domains
        - Pillars: domain-specific compositions branching UP from primitives, never merge
        - Orphans: unclassified observations that decay naturally
        - Epiphany: rare cross-domain read-only resonance detection

    Tag-aware branching: when an observation matches an existing pillar but carries
    a different tag, the engine branches — creating a new pillar sharing the same
    primitives under the new domain. Same roots, separate pillars.

    Usage:
        engine = SattvaEngine(dim=16)
        result = engine.observe(vector, tag='geometry', name='triangle')
        print(engine.status())
    """

    def __init__(self, dim, config=None):
        self.dim = dim
        c = config or {}
        self.sigmoid_k = c.get('sigmoid_k', 6.0)
        self.sigmoid_x0 = c.get('sigmoid_x0', 0.55)
        self.prim_match = c.get('prim_match', 0.30)
        self.pillar_match = c.get('pillar_match', 0.75)
        self.crystal_sim = c.get('crystal_sim', 0.60)
        self.crystal_min = c.get('crystal_min', 4)
        self.epiphany_cd = c.get('epiphany_cd', 35)
        self.epiphany_thresh = c.get('epiphany_thresh', 0.58)
        self.myel_rate = c.get('myel_rate', 0.05)
        self.myel_decay = c.get('myel_decay', 0.008)
        self.orphan_ttl = c.get('orphan_ttl', 25)

        # Coldness and pruning configuration
        self.cold_myel_thresh = c.get('cold_myel_thresh', 0.05)
        self.cold_total_thresh = c.get('cold_total_thresh', 3)
        self.cold_age_thresh = c.get('cold_age_thresh', 100)
        self.cold_sim_scale = c.get('cold_sim_scale', 0.5)

        self.primitives = []
        self.pillars = []
        self.orphans = []
        self.epiphanies = []
        self.step = 0
        self.last_epiphany = -999
        self.cost = 0

    def _sigmoid(self, d):
        return 1.0 / (1.0 + np.exp(self.sigmoid_k * (d - self.sigmoid_x0)))

    def cold_pillars(self,
                     myel_thresh=None,
                     total_thresh=None,
                     age_thresh=None):
        """
        Return indices of very cold pillars as a feedback signal.

        Cold = low myelination, low total activations, and sufficiently old.
        """
        myel_thresh = self.cold_myel_thresh if myel_thresh is None else myel_thresh
        total_thresh = self.cold_total_thresh if total_thresh is None else total_thresh
        age_thresh = self.cold_age_thresh if age_thresh is None else age_thresh

        cold = []
        for i, p in enumerate(self.pillars):
            if p['myel'] < myel_thresh and p['total'] < total_thresh and p['age'] > age_thresh:
                cold.append(i)
        return cold

    def observe(self, v, tag=None, name=None):
        """
        Feed one observation vector.

        Parameters
        ----------
        v : array-like
            Input vector (will be normalized).
        tag : any
            Domain / stream label (e.g., 'geometry', 'physics', 'email').
        name : any
            Optional human / symbolic label for this observation
            (e.g., 'triangle', 'Widget Co Deal 345').

        Returns
        -------
        dict describing what happened (pillar_match, new_pillar, branch_pillar, orphan)
        and any crystallization / epiphany event.
        """
        v = np.array(v, dtype=float)
        v = v / np.linalg.norm(v)
        self.step += 1
        self.cost += 1
        result = {'action': None}

        # Find best matching pillar overall and best same-tag pillar
        bi, bs = -1, 0.0
        bi_tag, bs_tag = -1, 0.0
        for i, p in enumerate(self.pillars):
            s = abs(np.dot(v, p['v']))

            # Cold-aware scaling: down-weight very cold pillars
            if (p['myel'] < self.cold_myel_thresh and
                p['total'] < self.cold_total_thresh and
                p['age'] > self.cold_age_thresh):
                s *= self.cold_sim_scale

            if s > bs:
                bs = s
                bi = i
            if s > bs_tag and p['tag'] == tag:
                bs_tag = s
                bi_tag = i

        # Prefer same-tag pillar match
        if bi_tag >= 0 and bs_tag > self.pillar_match:
            p = self.pillars[bi_tag]
            p['activations'] += 1
            p['total'] += 1
            lr = 0.005 * (1.0 - p['myel'])
            p['v'] = p['v'] + lr * (v - p['v'])
            p['v'] /= np.linalg.norm(p['v'])
            for pi in p['prim_ids']:
                if pi < len(self.primitives):
                    self.primitives[pi]['activations'] += 1
            self.cost += 1
            result = {
                'action': 'pillar_match',
                'pillar': bi_tag,
                'sim': float(bs_tag),
                'tag': tag,
            }

        elif bi >= 0 and bs > self.pillar_match:
            # Different tag, same pattern -> BRANCH: new pillar, shared primitives
            src = self.pillars[bi]
            self.pillars.append({
                'v': v.copy(),
                'prim_ids': list(src['prim_ids']),
                'tag': tag,
                'name': name,  # may be None
                'activations': 1,
                'total': 1,
                'myel': 0.0,
                'age': 0,
            })
            pid = len(self.pillars) - 1
            for pi in src['prim_ids']:
                if pi < len(self.primitives):
                    self.primitives[pi]['activations'] += 1
                    self.primitives[pi]['pillar_ids'].add(pid)
            self.cost += 3
            result = {
                'action': 'branch_pillar',
                'pillar': pid,
                'from': bi,
                'tag': tag,
            }

        else:
            # Decompose into known primitives
            mp = [(i, abs(np.dot(v, pr['v'])))
                  for i, pr in enumerate(self.primitives)]
            mp = [(i, s) for i, s in mp if s > self.prim_match]
            mp.sort(key=lambda x: -x[1])

            if mp:
                pids = [m[0] for m in mp[:4]]
                self.pillars.append({
                    'v': v.copy(),
                    'prim_ids': pids,
                    'tag': tag,
                    'name': name,  # may be None
                    'activations': 1,
                    'total': 1,
                    'myel': 0.0,
                    'age': 0,
                })
                pid = len(self.pillars) - 1
                for pi in pids:
                    self.primitives[pi]['activations'] += 1
                    self.primitives[pi]['pillar_ids'].add(pid)
                self.cost += 3
                result = {
                    'action': 'new_pillar',
                    'pillar': pid,
                    'prims': pids,
                    'tag': tag,
                }
            else:
                # Genuinely novel -> orphan (cheap, decays naturally)
                self.orphans.append({'v': v.copy(), 'age': 0, 'tag': tag})
                self.cost += 1
                result = {'action': 'orphan', 'tag': tag}

        crystal = self._try_crystallize()
        if crystal:
            result['crystallized'] = crystal

        epi = self._check_epiphany()
        if epi:
            result['epiphany'] = epi

        self._tick()
        return result

    def _try_crystallize(self):
        """Check if orphans cluster tightly enough to form a new primitive."""
        if len(self.orphans) < self.crystal_min:
            return None

        best_cl, best_s = None, 0
        limit = min(15, len(self.orphans))
        for i in range(limit):
            cl = [i]
            for j in range(i + 1, limit):
                if abs(np.dot(self.orphans[i]['v'],
                              self.orphans[j]['v'])) >= self.crystal_sim:
                    cl.append(j)
            if len(cl) >= self.crystal_min:
                pairs = [
                    (self.orphans[cl[a]]['v'], self.orphans[cl[b]]['v'])
                    for a in range(len(cl))
                    for b in range(a + 1, len(cl))
                ]
                if pairs:
                    avg = np.mean([abs(np.dot(a, b)) for a, b in pairs])
                    if avg > best_s:
                        best_cl = cl
                        best_s = avg

        if not best_cl:
            return None

        vecs = [self.orphans[i]['v'] for i in best_cl]
        centroid = np.mean(vecs, axis=0)
        centroid /= np.linalg.norm(centroid)

        # Check redundancy with existing primitives
        for pr in self.primitives:
            if abs(np.dot(centroid, pr['v'])) > 0.85:
                for idx in sorted(best_cl, reverse=True):
                    self.orphans.pop(idx)
                return None

        self.primitives.append({
            'v': centroid,
            'myel': 0.05,
            'activations': len(best_cl),
            'age': 0,
            'pillar_ids': set(),
        })
        self.cost += 5

        for idx in sorted(best_cl, reverse=True):
            self.orphans.pop(idx)

        return {
            'prim_id': len(self.primitives) - 1,
            'evidence': len(best_cl),
        }

    def _check_epiphany(self):
        """Cross-domain resonance detection. Read-only, never merges."""
        if (self.step - self.last_epiphany) < self.epiphany_cd:
            return None
        if len(self.pillars) < 4:
            return None

        groups = {}
        for i, p in enumerate(self.pillars):
            if p['tag'] is not None:
                groups.setdefault(p['tag'], []).append(i)

        tags = list(groups.keys())
        if len(tags) < 2:
            return None

        best = None
        for di in range(min(5, len(tags))):
            for dj in range(di + 1, min(5, len(tags))):
                s1 = np.random.choice(
                    groups[tags[di]],
                    min(3, len(groups[tags[di]])),
                    replace=False,
                )
                s2 = np.random.choice(
                    groups[tags[dj]],
                    min(3, len(groups[tags[dj]])),
                    replace=False,
                )
                for p1i in s1:
                    for p2i in s2:
                        p1 = self.pillars[p1i]
                        p2 = self.pillars[p2i]
                        d = 1.0 - abs(np.dot(p1['v'], p2['v']))
                        sh = set(p1['prim_ids']) & set(p2['prim_ids'])
                        r = (
                            self._sigmoid(d)
                            * min(1.0, (p1['total'] + p2['total']) / 8.0)
                            * (1 + 0.3 * len(sh))
                        )
                        if r > self.epiphany_thresh:
                            mag = d * r * (p1['total'] + p2['total'])
                            if not best or mag > best['magnitude']:
                                best = {
                                    'step': self.step,
                                    'tags': [tags[di], tags[dj]],
                                    'pillars': [p1i, p2i],
                                    'pillar_names': [
                                        p1.get('name'),
                                        p2.get('name'),
                                    ],
                                    'shared_prims': len(sh),
                                    'resonance': float(r),
                                    'magnitude': float(mag),
                                }
                self.cost += 2

        if best:
            # Attach a UUID so all items participating in this epiphany
            # can be correlated downstream.
            best['id'] = str(uuid.uuid4())
            self.epiphanies.append(best)
            self.last_epiphany = self.step
            self.cost += 8
        return best

    def _tick(self):
        """Myelination, decay, orphan cleanup."""
        for pr in self.primitives:
            pr['age'] += 1
            if pr['activations'] > 0:
                pr['myel'] = min(
                    1.0,
                    pr['myel'] + self.myel_rate * pr['activations'],
                )
            else:
                pr['myel'] = max(0.0, pr['myel'] - self.myel_decay)
            pr['activations'] = 0

        for p in self.pillars:
            p['age'] += 1
            if p['activations'] > 0:
                p['myel'] = min(
                    1.0,
                    p['myel'] + self.myel_rate * 0.5 * p['activations'],
                )
            else:
                p['myel'] = max(
                    0.0,
                    p['myel'] - self.myel_decay * 0.5,
                )
            p['activations'] = 0

        self.orphans = [o for o in self.orphans if o['age'] < self.orphan_ttl]
        for o in self.orphans:
            o['age'] += 1
        self.orphans = [o for o in self.orphans if o['age'] < self.orphan_ttl]

    def prune_pillars(self,
                      min_myel=0.0,
                      min_total=0,
                      max_per_tag=None,
                      protect_epiphany=False):
        """
        Drop cold/weak pillars; optional cap per tag.

        By default, does NOT use epiphany protection because epiphany
        records don't yet store pillar indices beyond what we log.
        """
        protected = set()
        if protect_epiphany:
            for e in self.epiphanies:
                for idx in e.get('pillars', []):
                    protected.add(idx)

        keep = []
        for i, p in enumerate(self.pillars):
            if i in protected:
                keep.append((i, p))
                continue
            # Drop only if below BOTH minimums
            if p['myel'] < min_myel and p['total'] < min_total:
                continue
            keep.append((i, p))

        if max_per_tag is not None:
            by_tag = {}
            for i, p in keep:
                by_tag.setdefault(p['tag'], []).append((i, p))
            new_keep = []
            for tag, items in by_tag.items():
                items.sort(key=lambda x: x[1]['myel'], reverse=True)
                new_keep.extend(items[:max_per_tag])
            keep = new_keep

        new_pillars = []
        index_map = {}
        for old_i, p in keep:
            index_map[old_i] = len(new_pillars)
            new_pillars.append(p)
        self.pillars = new_pillars

        for pr in self.primitives:
            pr['pillar_ids'] = {index_map[i] for i in pr['pillar_ids'] if i in index_map}

    def prune_primitives(self, min_myel=0.0, require_pillars=True):
        """Drop primitives that are unused and unmyelinated."""
        keep = []
        for i, pr in enumerate(self.primitives):
            if pr['myel'] < min_myel:
                continue
            if require_pillars and not pr['pillar_ids']:
                continue
            keep.append((i, pr))

        index_map = {}
        new_prims = []
        for old_i, pr in keep:
            index_map[old_i] = len(new_prims)
            pr['pillar_ids'] = set()
            new_prims.append(pr)
        self.primitives = new_prims

        # Remap primitive ids in pillars and rebuild pillar_ids
        for p_idx, p in enumerate(self.pillars):
            new_prim_ids = []
            for old_pi in p['prim_ids']:
                if old_pi in index_map:
                    new_pi = index_map[old_pi]
                    new_prim_ids.append(new_pi)
                    self.primitives[new_pi]['pillar_ids'].add(p_idx)
            p['prim_ids'] = new_prim_ids

    def status(self):
        """Return current engine state summary."""
        return {
            'step': self.step,
            'primitives': len(self.primitives),
            'pillars': len(self.pillars),
            'orphans': len(self.orphans),
            'epiphanies': len(self.epiphanies),
            'cost': self.cost,
            'prim_myel': [round(p['myel'], 3) for p in self.primitives],
        }
