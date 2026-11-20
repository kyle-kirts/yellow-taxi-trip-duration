const METRES_PER_LON = 84300;
const METRES_PER_LAT = 111200;
const COST_PER_METRE = 36 / 110;

class MinHeap {
  constructor() {
    this.items = [];
  }

  push(entry) {
    const items = this.items;
    let index = items.length;
    items.push(entry);
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (items[parent][2] <= entry[2]) break;
      items[index] = items[parent];
      index = parent;
    }
    items[index] = entry;
  }

  pop() {
    const items = this.items;
    const first = items[0];
    const last = items.pop();
    if (!items.length) return first;
    let index = 0;
    while (index * 2 + 1 < items.length) {
      let child = index * 2 + 1;
      if (child + 1 < items.length && items[child + 1][2] < items[child][2]) child += 1;
      if (last[2] <= items[child][2]) break;
      items[index] = items[child];
      index = child;
    }
    items[index] = last;
    return first;
  }
}

function indexGraph(data) {
  const points = Int32Array.from(data.points);
  const edges = Int32Array.from(data.links);
  const size = points.length / 2;
  const adjacency = new Uint32Array(size + 1);
  const inflow = new Uint32Array(size);
  for (let i = 0; i < edges.length; i += 4) {
    adjacency[edges[i] + 1] += 1;
    inflow[edges[i + 1]] += 1;
  }
  for (let i = 1; i <= size; i += 1) adjacency[i] += adjacency[i - 1];
  const outgoing = new Uint32Array(edges.length / 4);
  const cursor = adjacency.slice();
  for (let i = 0; i < edges.length; i += 4) outgoing[cursor[edges[i]]++] = i;
  const turns = new Map();
  for (const [node, from, to, kind] of data.turns) {
    if (!turns.has(node)) turns.set(node, new Map());
    const turn = turns.get(node);
    if (!turn.has(from)) turn.set(from, []);
    turn.get(from).push({ to, kind });
  }
  return { points, edges, size, adjacency, outgoing, inflow, turns };
}

async function loadGraph() {
  const response = await fetch("nyc-road-graph.json.gz");
  if (!response.ok) throw new Error("Road network unavailable");
  const plain = response.body.pipeThrough(new DecompressionStream("gzip"));
  return indexGraph(await new Response(plain).json());
}

function snap(graph, [lon, lat], mode) {
  const { points, size, adjacency, inflow } = graph;
  const candidates = [];
  for (let node = 0; node < size; node += 1) {
    const routable = mode === "out" ? adjacency[node] < adjacency[node + 1] : inflow[node] > 0;
    if (!routable) continue;
    const dx = (points[2 * node] / 1e5 - lon) * METRES_PER_LON;
    const dy = (points[2 * node + 1] / 1e5 - lat) * METRES_PER_LAT;
    const distance = dx * dx + dy * dy;
    if (candidates.length < 8 || distance < candidates.at(-1).distance) {
      candidates.push({ node, distance });
      candidates.sort((a, b) => a.distance - b.distance);
      if (candidates.length > 8) candidates.pop();
    }
  }
  return candidates.map(({ node, distance }) => ({ node, metres: Math.sqrt(distance) }));
}

function pathBetween(graph, start, end) {
  const { size, points, edges, adjacency, outgoing, turns } = graph;
  const states = size + outgoing.length;
  graph.search ??= {
    distances: new Float64Array(states),
    previousEdge: new Int32Array(states),
    parents: new Int32Array(states),
    seen: new Uint32Array(states),
    round: 0,
  };
  const { distances, previousEdge, parents, seen } = graph.search;
  const round = ++graph.search.round;
  const heap = new MinHeap();

  function estimate(node) {
    return Math.hypot(
      (points[2 * end] - points[2 * node]) * METRES_PER_LON / 1e5,
      (points[2 * end + 1] - points[2 * node + 1]) * METRES_PER_LAT / 1e5,
    ) * COST_PER_METRE;
  }

  distances[start] = 0;
  previousEdge[start] = -1;
  parents[start] = -1;
  seen[start] = round;
  heap.push([start, 0, estimate(start)]);
  while (heap.items.length) {
    const [state, cost] = heap.pop();
    if (cost !== distances[state]) continue;
    const incoming = previousEdge[state];
    const node = state < size ? state : edges[incoming + 1];
    if (node === end) {
      const path = [];
      for (let cursor = state; parents[cursor] !== -1; cursor = parents[cursor]) path.push(previousEdge[cursor]);
      return path.reverse();
    }
    const inboundWay = incoming === -1 ? null : edges[incoming + 3];
    const rules = turns.get(node)?.get(inboundWay) || [];
    for (let i = adjacency[node]; i < adjacency[node + 1]; i += 1) {
      const edge = outgoing[i];
      const next = edges[edge + 1];
      const way = edges[edge + 3];
      if (rules.some((rule) => rule.kind === 1 ? way !== rule.to : way === rule.to && (rule.kind === 0 || next === edges[incoming]))) continue;
      const targetState = turns.has(next) ? size + edge / 4 : next;
      const candidate = cost + edges[edge + 2];
      if (seen[targetState] === round && candidate >= distances[targetState]) continue;
      seen[targetState] = round;
      distances[targetState] = candidate;
      previousEdge[targetState] = edge;
      parents[targetState] = state;
      heap.push([targetState, candidate, candidate + estimate(next)]);
    }
  }
  return null;
}

function chooseEndpoints(graph, from, to) {
  const originOptions = snap(graph, from, "out");
  const destinationOptions = snap(graph, to, "in");
  // Try nearby network points if the closest road lies on an isolated segment.
  const pairs = originOptions.flatMap((a) => destinationOptions.map((b) => ({ a, b, distance: a.metres + b.metres })))
    .filter(({ a, b }) => a.node !== b.node && a.metres < 1000 && b.metres < 1000)
    .sort((a, b) => a.distance - b.distance);
  for (const { a, b } of pairs.slice(0, 10)) {
    const path = pathBetween(graph, a.node, b.node);
    if (path?.length) return { start: a.node, path };
  }
  return null;
}

function describeRoad(graph, { start, path }) {
  const { edges, points } = graph;
  const stations = [start, ...path.map((edge) => edges[edge + 1])];
  const coordinates = stations.map((node) => [points[node * 2] / 1e5, points[node * 2 + 1] / 1e5]);
  let metres = 0;
  for (let i = 1; i < coordinates.length; i += 1) {
    metres += Math.hypot(
      (coordinates[i][0] - coordinates[i - 1][0]) * METRES_PER_LON,
      (coordinates[i][1] - coordinates[i - 1][1]) * METRES_PER_LAT,
    );
  }
  return { coordinates, miles: metres / 1609.344 };
}

let graphPromise;
self.addEventListener("message", async ({ data }) => {
  try {
    graphPromise ??= loadGraph();
    const graph = await graphPromise;
    const endpoints = chooseEndpoints(graph, data.from, data.to);
    const route = endpoints ? describeRoad(graph, endpoints) : null;
    self.postMessage({ id: data.id, route });
  } catch (error) {
    graphPromise = null;
    self.postMessage({ id: data.id, error: error.message });
  }
});
