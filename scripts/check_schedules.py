#!/usr/bin/env python3
"""CPU models of the advanced scheduling invariants; no GPU validation claim.

Models deliberately separate TMA issue/completion, MMA issue/completion, warp
releases, partial publication and output reads. Run from any directory. They
cannot validate PTX semantics, CUDA memory ordering, or numerical GEMM accuracy.
"""
from collections import deque
import random


SLOTS = (2, 3, 4, 5, 6, 8)


def consumer_program(counts, overlap):
    tick = 0
    for count in counts:
        for step in range(count):
            yield "full", tick
            yield "mma", tick
            if not overlap or step:
                yield "wait", int(overlap)
                yield "release", tick - int(overlap)
            tick += 1
        if overlap:
            yield "wait", 0
            yield "release", tick - 1
        yield "store", tick


def ring_trial(slots, counts, ranks, overlap, rng):
    """A single ring, optionally replicated in a two-CTA multicast cluster."""
    total = sum(counts)
    producer = [0] * ranks
    # Last generation with ALL consumer-warp releases; -1 means initially free.
    released = [-1] * slots
    current = [[None] * slots for _ in range(ranks)]
    armed, copied, arrivals = set(), set(), {}
    transfers, warp_releases = [], []
    pending = [deque() for _ in range(ranks)]
    safe = [set() for _ in range(ranks)]
    programs = [iter(consumer_program(counts, overlap)) for _ in range(ranks)]
    operation = [next(p) for p in programs]
    completed = [False] * ranks
    multicast = set()
    events = 0
    while not (all(completed) and not transfers and not warp_releases and min(released) >= total - slots):
        actions = []
        for rank in range(ranks):
            t = producer[rank]
            if t < total and (t < slots or released[t % slots] >= t - slots):
                actions.append(("produce", rank))
            if pending[rank]:
                first = next((i for i, (_, done) in enumerate(pending[rank]) if not done), None)
                if first is not None:
                    actions.append(("finish_mma", rank, first))
            if completed[rank]:
                continue
            op, value = operation[rank]
            if op == "full":
                ready = (rank, value, "a") in copied and (rank, value, "b") in copied
            elif op == "wait":
                ready = all(done for _, done in list(pending[rank])[:max(0, len(pending[rank]) - value)])
            else:
                ready = True
            if ready:
                actions.append(("consume", rank))
        if ranks == 2:
            for t in range(max(0, min(producer) - slots), min(producer)):
                if t not in multicast and all((r, t) in armed for r in range(ranks)):
                    actions.append(("multicast", t))
        actions += [("transfer", i) for i in range(len(transfers))]
        actions += [("release_warp", i) for i in range(len(warp_releases))]
        assert actions, ("deadlock", slots, counts, ranks, producer, operation)
        event = rng.choice(actions)
        events += 1
        assert events < 100000
        kind, index, *extra = event
        if kind == "produce":
            t = producer[index]
            # No still-live async MMA may hold the overwritten operand.
            for queue in pending:
                assert all(old % slots != t % slots or old == t for old, done in queue if not done)
            current[index][t % slots] = t
            armed.add((index, t))
            transfers.append([(index, t, "b")])
            if ranks == 1:
                transfers.append([(index, t, "a")])
            producer[index] += 1
        elif kind == "multicast":
            t = index
            assert all((r, t) in armed for r in range(ranks))
            transfers.append([(r, t, "a") for r in range(ranks)])
            multicast.add(t)
        elif kind == "transfer":
            for rank, t, operand in transfers.pop(index):
                assert current[rank][t % slots] == t
                copied.add((rank, t, operand))
        elif kind == "finish_mma":
            i = extra[0]
            t, _ = pending[index][i]
            assert current[index][t % slots] == t
            pending[index][i] = (t, True)
        elif kind == "release_warp":
            t, rank, warp = warp_releases.pop(index)
            arrivals.setdefault(t, set()).add((rank, warp))
            # Each warp signals both replicas, which have identical lifetimes.
            if len(arrivals[t]) == 4 * ranks:
                released[t % slots] = t
        else:
            op, value = operation[index]
            if op == "mma":
                assert current[index][value % slots] == value
                pending[index].append((value, False))
                assert len(pending[index]) <= (2 if overlap else 1)
            elif op == "wait":
                while len(pending[index]) > value:
                    t, done = pending[index].popleft()
                    assert done
                    safe[index].add(t)
            elif op == "release":
                assert value in safe[index], "release before wait proves operand completion"
                safe[index].remove(value)
                warp_releases.extend((value, index, w) for w in range(4))
            elif op == "store":
                assert not pending[index], "epilogue reads an outstanding accumulator"
            operation[index] = next(programs[index], None)
            completed[index] = operation[index] is None
    assert all(t == total for t in producer)
    assert all(len(arrivals[t]) == 4 * ranks for t in range(total))


def segments(tiles, kg, workers):
    units = tiles * kg
    result = []
    for w in range(workers):
        cursor, end = units * w // workers, units * (w + 1) // workers
        while cursor < end:
            tile, first = divmod(cursor, kg)
            count = min(end - cursor, kg - first)
            result.append((w, tile, first, count))
            cursor += count
    return result


def stream_k_trial(tiles, kg, workers, rng):
    units = tiles * kg
    work = segments(tiles, kg, workers)
    by_tile = [[] for _ in range(tiles)]
    boundaries = [[] for _ in range(workers)]
    previous = 0
    for w, tile, first, count in work:
        assert tile * kg + first == previous
        previous += count
        by_tile[tile].append((w, first, count))
        if first or count != kg:
            boundaries[w].append(tile)
    assert previous == units
    assert all(len(t) <= 2 for t in boundaries)
    slots = {}
    for w, tile, first, count in work:
        if first or count != kg:
            side = int(tile != (units * w // workers) // kg)
            slot = 2 * w + side
            assert slot not in slots
            slots[slot] = tile
    counters = [0] * tiles
    workspace = {}
    partials = [(w, t, first, count) for w, t, first, count in work if first or count != kg]
    for replay in range(3):
        output = {}
        shuffled = partials.copy()
        rng.shuffle(shuffled)
        for w, tile, first, count in shuffled:
            side = int(tile != (units * w // workers) // kg)
            workspace[2 * w + side] = (replay, float(count) + w * 0.125)
            start = ((tile * kg + 1) * workers - 1) // units
            stop = (((tile + 1) * kg) * workers - 1) // units
            assert list(range(start, stop + 1)) == [owner for owner, _, _ in by_tile[tile]]
            counters[tile] += 1
            if counters[tile] == stop - start + 1:
                values = []
                for owner in range(start, stop + 1):
                    slot = 2 * owner + int(tile != (units * owner // workers) // kg)
                    generation, value = workspace[slot]
                    assert generation == replay, "stale partial from prior graph replay"
                    values.append(value)
                assert tile not in output
                output[tile] = sum(values)  # Fixed owner order, never arrival order.
                counters[tile] = 0
        assert not any(counters)
        for tile, owners in enumerate(by_tile):
            if len(owners) == 1 and owners[0][1:] == (0, kg):
                assert tile not in output
            else:
                assert output[tile] == sum(float(count) + owner * 0.125 for owner, _, count in owners)


def pingpong_trial(slots, jobs, workers, steps, rng):
    written = set()
    for worker in range(workers):
        task = [list(range(worker * 2 + g, jobs, workers * 2)) for g in range(2)]
        # Model the ordered math token and independently completing TMA reads
        # and global writes. A group cannot overwrite a shared buffer still read.
        turn, math, read, global_stores = 0, [None, None], [None, None], []
        produced, consumed = [0, 0], [0, 0]
        done = [0, 0]
        while any(task) or any(x is not None for x in math + read) or global_stores:
            actions = []
            for g in range(2):
                demand = (done[g] + len(task[g])) * steps
                if produced[g] < demand and produced[g] - consumed[g] < slots:
                    actions.append(("prefetch", g))
                if task[g] and turn == g and math[g] is None:
                    if produced[g] > consumed[g]:
                        actions.append(("mma", g))
                if math[g] is not None and read[g] is None:
                    actions.append(("store", g))
                if read[g] is not None:
                    actions.append(("read_done", g))
            actions += [("write_done", i) for i in range(len(global_stores))]
            assert actions, "ping-pong deadlock"
            action, g = rng.choice(actions)
            if action == "prefetch": produced[g] += 1
            elif action == "mma":
                consumed[g] += 1
                if consumed[g] % steps == 0:
                    math[g] = task[g].pop(0)
                    done[g] += 1
                    turn = 1 - g
            elif action == "store":
                read[g], math[g] = math[g], None
            elif action == "read_done":
                global_stores.append(read[g]); read[g] = None
            else:
                job = global_stores.pop(g)
                assert job not in written
                written.add(job)
        assert produced == consumed
    assert written == set(range(jobs))


def layout_checks():
    for bn in (16, 32, 64, 128, 256):
        pairs = set()
        for tid in range(128):
            row = (tid // 32) * 16 + (tid % 32) // 4
            for j in range(bn // 8):
                col = j * 8 + (tid % 4) * 2
                for r in (row, row + 8):
                    assert (r, col) not in pairs
                    assert (r * bn + col) * 4 % 8 == 0
                    pairs.add((r, col))
        assert len(pairs) == 64 * bn // 2
        for batch in (1, 2):
            for rows in (64, bn):
                for atom in range(batch):
                    for row in range(rows):
                        for k in (0, 7, 16, 63):
                            # 3D tensor coordinates [k,row,k_atom] recover the
                            # original [row,K] global address with no repack.
                            k0 = 17
                            assert k + row * 8192 + (k0 + atom) * 64 == row * 8192 + k0 * 64 + atom * 64 + k
                            shared = atom * rows * 128 + row * 128 + 2 * k
                            swizzled = shared ^ ((shared & 0x380) >> 3)
                            assert 0 <= swizzled < batch * rows * 128
                for slots in SLOTS:
                    assert (batch * rows * 128 * slots) % 1024 == 0


def main():
    rng = random.Random(20261124)
    ring_count = stream_count = ping_count = 0
    for slots in SLOTS:
        for counts in ([1], [1, 1, 1], [2, 3, 1, 7], [17, 1, 2, 9], [32, 32, 32]):
            for ranks in (1, 2):
                for overlap in (False, True):
                    for _ in range(4):
                        ring_trial(slots, counts, ranks, overlap, rng)
                        ring_count += 1
    for tiles in (1, 3, 16, 32, 64, 128, 256):
        for kg in (1, 2, 7, 64, 128):
            units = tiles * kg
            choices = set(range(1, min(units, 129) + 1)) | {units, max(1, units - 1)}
            choices |= {rng.randint(1, units) for _ in range(8)}
            for workers in sorted(choices):
                stream_k_trial(tiles, kg, workers, rng)
                stream_count += 1
    for slots in SLOTS:
        for jobs in (1, 2, 3, 17, 64):
            for workers in sorted({1, min(3, (jobs + 1) // 2), (jobs + 1) // 2}):
                for steps in (1, 2, 7):
                    pingpong_trial(slots, jobs, workers, steps, rng)
                    ping_count += 1
    layout_checks()
    print(f"PASS CPU models: {ring_count} ring/multicast schedules, {stream_count} Stream-K partitions x 3 replays, {ping_count} ping-pong schedules; layouts/alignment")
    print("GPU correctness, CUDA memory ordering, and performance still require hardware validation.")


if __name__ == "__main__":
    main()
