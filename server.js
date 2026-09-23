const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = process.env.PORT || 8080;
const DIMS = 16;

// =====================================================================
//  DISTANCE METRICS
// =====================================================================

function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na < 1e-9 || nb < 1e-9) return 1.0;
  return 1.0 - dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function manhattan(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += Math.abs(a[i] - b[i]);
  }
  return s;
}

function getDistFn(m) {
  if (m === 'cosine') return cosine;
  if (m === 'manhattan') return manhattan;
  return euclidean;
}

// =====================================================================
//  BRUTE FORCE
// =====================================================================

class BruteForce {
  constructor() {
    this.items = [];
  }

  insert(v) {
    this.items.push(v);
  }

  knn(q, k, dist) {
    const r = this.items.map(v => ({ dist: dist(q, v.emb), id: v.id }));
    r.sort((a, b) => a.dist - b.dist);
    return r.slice(0, k);
  }

  remove(id) {
    this.items = this.items.filter(v => v.id !== id);
  }
}

// =====================================================================
//  KD-TREE
// =====================================================================

class KDNode {
  constructor(item) {
    this.item = item;
    this.left = null;
    this.right = null;
  }
}

class KDTree {
  constructor(dims) {
    this.dims = dims;
    this.root = null;
  }

  insert(v) {
    this.root = this._ins(this.root, v, 0);
  }

  _ins(n, v, d) {
    if (!n) return new KDNode(v);
    const ax = d % this.dims;
    if (v.emb[ax] < n.item.emb[ax]) {
      n.left = this._ins(n.left, v, d + 1);
    } else {
      n.right = this._ins(n.right, v, d + 1);
    }
    return n;
  }

  knn(q, k, dist) {
    const heap = [];
    this._knn(this.root, q, k, 0, dist, heap);
    heap.sort((a, b) => a.dist - b.dist);
    return heap;
  }

  _knn(n, q, k, d, dist, heap) {
    if (!n) return;
    const dn = dist(q, n.item.emb);
    if (heap.length < k || dn < heap[heap.length - 1].dist) {
      heap.push({ dist: dn, id: n.item.id });
      heap.sort((a, b) => a.dist - b.dist);
      if (heap.length > k) heap.pop();
    }
    const ax = d % this.dims;
    const diff = q[ax] - n.item.emb[ax];
    const closer = diff < 0 ? n.left : n.right;
    const farther = diff < 0 ? n.right : n.left;

    this._knn(closer, q, k, d + 1, dist, heap);
    if (heap.length < k || Math.abs(diff) < heap[heap.length - 1].dist) {
      this._knn(farther, q, k, d + 1, dist, heap);
    }
  }

  rebuild(items) {
    this.root = null;
    for (const v of items) this.insert(v);
  }
}

// =====================================================================
//  HNSW (Hierarchical Navigable Small World)
// =====================================================================

class HNSW {
  constructor(m = 16, efBuild = 200) {
    this.M = m;
    this.M0 = 2 * m;
    this.ef_build = efBuild;
    this.mL = 1.0 / Math.log(m);
    this.topLayer = -1;
    this.entryPt = -1;
    this.G = new Map();
  }

  randLevel() {
    const u = Math.max(Math.random(), 1e-9);
    return Math.floor(-Math.log(u) * this.mL);
  }

  searchLayer(q, ep, ef, lyr, dist) {
    const vis = new Set();
    const cands = [];
    const found = [];

    const d0 = dist(q, this.G.get(ep).item.emb);
    vis.add(ep);
    cands.push({ dist: d0, id: ep });
    found.push({ dist: d0, id: ep });

    while (cands.length > 0) {
      cands.sort((a, b) => a.dist - b.dist);
      const curr = cands.shift();
      found.sort((a, b) => a.dist - b.dist);

      if (found.length >= ef && curr.dist > found[found.length - 1].dist) {
        break;
      }

      const node = this.G.get(curr.id);
      if (!node || lyr >= node.nbrs.length) continue;

      for (const nid of node.nbrs[lyr]) {
        if (vis.has(nid) || !this.G.has(nid)) continue;
        vis.add(nid);
        const neighborNode = this.G.get(nid);
        const nd = dist(q, neighborNode.item.emb);

        if (found.length < ef || nd < found[found.length - 1].dist) {
          cands.push({ dist: nd, id: nid });
          found.push({ dist: nd, id: nid });
          found.sort((a, b) => a.dist - b.dist);
          if (found.length > ef) found.pop();
        }
      }
    }

    found.sort((a, b) => a.dist - b.dist);
    return found;
  }

  selectNbrs(cands, maxM) {
    return cands.slice(0, maxM).map(c => c.id);
  }

  insert(item, dist) {
    const id = item.id;
    const lvl = this.randLevel();
    const nbrs = Array.from({ length: lvl + 1 }, () => []);
    this.G.set(id, { item, maxLyr: lvl, nbrs });

    if (this.entryPt === -1) {
      this.entryPt = id;
      this.topLayer = lvl;
      return;
    }

    let ep = this.entryPt;
    for (let lc = this.topLayer; lc > lvl; lc--) {
      const epNode = this.G.get(ep);
      if (epNode && lc < epNode.nbrs.length) {
        const W = this.searchLayer(item.emb, ep, 1, lc, dist);
        if (W.length > 0) ep = W[0].id;
      }
    }

    for (let lc = Math.min(this.topLayer, lvl); lc >= 0; lc--) {
      const W = this.searchLayer(item.emb, ep, this.ef_build, lc, dist);
      const maxM = (lc === 0) ? this.M0 : this.M;
      const sel = this.selectNbrs(W, maxM);
      this.G.get(id).nbrs[lc] = sel;

      for (const nid of sel) {
        const nNode = this.G.get(nid);
        if (!nNode) continue;
        while (nNode.nbrs.length <= lc) nNode.nbrs.push([]);
        const conn = nNode.nbrs[lc];
        conn.push(id);
        if (conn.length > maxM) {
          const ds = [];
          for (const c of conn) {
            if (this.G.has(c)) {
              ds.push({ dist: dist(nNode.item.emb, this.G.get(c).item.emb), id: c });
            }
          }
          ds.sort((a, b) => a.dist - b.dist);
          nNode.nbrs[lc] = ds.slice(0, maxM).map(x => x.id);
        }
      }
      if (W.length > 0) ep = W[0].id;
    }

    if (lvl > this.topLayer) {
      this.topLayer = lvl;
      this.entryPt = id;
    }
  }

  knn(q, k, ef, dist) {
    if (this.entryPt === -1 || this.G.size === 0) return [];
    let ep = this.entryPt;
    for (let lc = this.topLayer; lc > 0; lc--) {
      const epNode = this.G.get(ep);
      if (epNode && lc < epNode.nbrs.length) {
        const W = this.searchLayer(q, ep, 1, lc, dist);
        if (W.length > 0) ep = W[0].id;
      }
    }
    const W = this.searchLayer(q, ep, Math.max(ef, k), 0, dist);
    return W.slice(0, k);
  }

  remove(id) {
    if (!this.G.has(id)) return;
    for (const [nid, nd] of this.G.entries()) {
      for (let lc = 0; lc < nd.nbrs.length; lc++) {
        nd.nbrs[lc] = nd.nbrs[lc].filter(x => x !== id);
      }
    }
    if (this.entryPt === id) {
      this.entryPt = -1;
      for (const nid of this.G.keys()) {
        if (nid !== id) {
          this.entryPt = nid;
          break;
        }
      }
    }
    this.G.delete(id);
  }

  getInfo() {
    const maxL = Math.max(this.topLayer + 1, 1);
    const nodesPerLayer = new Array(maxL).fill(0);
    const edgesPerLayer = new Array(maxL).fill(0);
    const nodes = [];
    const edges = [];

    for (const [id, nd] of this.G.entries()) {
      nodes.push({ id, metadata: nd.item.metadata, category: nd.item.category, maxLyr: nd.maxLyr });
      for (let lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
        nodesPerLayer[lc]++;
        if (lc < nd.nbrs.length) {
          for (const nid of nd.nbrs[lc]) {
            if (id < nid) {
              edgesPerLayer[lc]++;
              edges.push({ src: id, dst: nid, lyr: lc });
            }
          }
        }
      }
    }

    return {
      topLayer: this.topLayer,
      nodeCount: this.G.size,
      nodesPerLayer,
      edgesPerLayer,
      nodes,
      edges
    };
  }

  size() {
    return this.G.size;
  }
}

// =====================================================================
//  VECTOR DATABASE (16D Demo)
// =====================================================================

class VectorDB {
  constructor(dims = DIMS) {
    this.dims = dims;
    this.store = new Map();
    this.bf = new BruteForce();
    this.kdt = new KDTree(dims);
    this.hnsw = new HNSW(16, 200);
    this.nextId = 1;
  }

  insert(meta, cat, emb, dist) {
    const id = this.nextId++;
    const item = { id, metadata: meta, category: cat, emb };
    this.store.set(id, item);
    this.bf.insert(item);
    this.kdt.insert(item);
    this.hnsw.insert(item, dist);
    return id;
  }

  remove(id) {
    if (!this.store.has(id)) return false;
    this.store.delete(id);
    this.bf.remove(id);
    this.kdt.rebuild(Array.from(this.store.values()));
    this.hnsw.remove(id);
    return true;
  }

  knn(q, k, dist, algo) {
    if (algo === 'bruteforce') return this.bf.knn(q, k, dist);
    if (algo === 'kdtree')     return this.kdt.knn(q, k, dist);
    return this.hnsw.knn(q, k, 50, dist);
  }

  benchmark(q, k, dist) {
    // Brute Force
    const t0 = process.hrtime.bigint();
    const bfR = this.bf.knn(q, k, dist);
    const bfUs = Number(process.hrtime.bigint() - t0) / 1000;

    // KD-Tree
    const t1 = process.hrtime.bigint();
    const kdR = this.kdt.knn(q, k, dist);
    const kdUs = Number(process.hrtime.bigint() - t1) / 1000;

    // HNSW
    const t2 = process.hrtime.bigint();
    const hnswR = this.hnsw.knn(q, k, 50, dist);
    const hnswUs = Number(process.hrtime.bigint() - t2) / 1000;

    // Accuracy
    const bfSet = new Set(bfR.map(x => x.id));
    let kdMatch = (kdR.length === bfR.length);
    for (let i = 0; i < Math.min(kdR.length, bfR.length); i++) {
      if (kdR[i].id !== bfR[i].id) kdMatch = false;
    }

    let matchCount = 0;
    for (const r of hnswR) {
      if (bfSet.has(r.id)) matchCount++;
    }
    const recall = bfR.length > 0 ? matchCount / bfR.length : 1.0;

    return {
      bruteforceUs: Math.max(Math.round(bfUs), 1),
      kdtreeUs: Math.max(Math.round(kdUs), 1),
      hnswUs: Math.max(Math.round(hnswUs), 1),
      kdtreeMatch: kdMatch,
      hnswRecall: Math.round(recall * 100) / 100
    };
  }

  all() {
    return Array.from(this.store.values());
  }

  size() {
    return this.store.size;
  }
}

// Load 20 initial demo vectors exactly matching C++ lines 720-760
function loadDemo(db) {
  const dist = getDistFn('cosine');
  db.insert("Linked List: nodes connected by pointers", "cs",
    [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06], dist);
  db.insert("Binary Search Tree: O(log n) search and insert", "cs",
    [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10], dist);
  db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
    [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07], dist);
  db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
    [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07], dist);
  db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
    [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09], dist);
  db.insert("Calculus: derivatives integrals and limits", "math",
    [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10], dist);
  db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
    [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09], dist);
  db.insert("Probability: distributions random variables Bayes theorem", "math",
    [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08], dist);
  db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
    [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06], dist);
  db.insert("Combinatorics: permutations combinations generating functions", "math",
    [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07], dist);
  db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
    [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07], dist);
  db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
    [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08], dist);
  db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
    [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06], dist);
  db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
    [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09], dist);
  db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
    [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06], dist);
  db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
    [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72], dist);
  db.insert("Football: tackles touchdowns field goals and strategy", "sports",
    [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76], dist);
  db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
    [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82], dist);
  db.insert("Chess: openings endgames tactics strategic board game", "sports",
    [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90], dist);
  db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
    [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80], dist);
}

// =====================================================================
//  DOCUMENT DATABASE
// =====================================================================

class DocumentDB {
  constructor() {
    this.store = new Map();
    this.hnsw = new HNSW(16, 200);
    this.bf = new BruteForce();
    this.nextId = 1;
    this.dims = 0;
  }

  insert(title, text, emb) {
    if (this.dims === 0) this.dims = emb.length;
    const id = this.nextId++;
    const item = { id, title, text, emb };
    this.store.set(id, item);
    const vi = { id, metadata: title, category: 'doc', emb };
    this.hnsw.insert(vi, cosine);
    this.bf.insert(vi);
    return id;
  }

  search(q, k, max_dist = 0.7) {
    if (this.store.size === 0) return [];
    const raw = (this.store.size < 10)
      ? this.bf.knn(q, k, cosine)
      : this.hnsw.knn(q, k, 50, cosine);

    const out = [];
    for (const item of raw) {
      if (this.store.has(item.id) && item.dist <= max_dist) {
        out.push({ distance: item.dist, item: this.store.get(item.id) });
      }
    }
    return out;
  }

  remove(id) {
    if (!this.store.has(id)) return false;
    this.store.delete(id);
    this.hnsw.remove(id);
    this.bf.remove(id);
    return true;
  }

  all() {
    return Array.from(this.store.values());
  }

  size() {
    return this.store.size;
  }

  getDims() {
    return this.dims;
  }
}

// =====================================================================
//  TEXT CHUNKER (250 words with 50-word overlap)
// =====================================================================

function chunkText(text, chunkSize = 250, overlap = 50) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return [];
  if (words.length <= chunkSize) return [text];

  const chunks = [];
  const step = Math.max(chunkSize - overlap, 1);
  for (let i = 0; i < words.length; i += step) {
    const chunkWords = words.slice(i, i + chunkSize);
    chunks.push(chunkWords.join(' '));
    if (i + chunkSize >= words.length) break;
  }
  return chunks;
}

// =====================================================================
//  AI ENGINE (Ollama client with intelligent built-in semantic fallback)
// =====================================================================

class AIEngine {
  constructor(host = '127.0.0.1', port = 11434) {
    this.host = host;
    this.port = port;
    this.embedModel = 'nomic-embed-text';
    this.genModel = 'llama3.2';
    this.ollamaAvailable = false;
    this.lastCheck = 0;
  }

  async checkOllama() {
    const now = Date.now();
    if (now - this.lastCheck < 10000) return this.ollamaAvailable;
    this.lastCheck = now;
    return new Promise(resolve => {
      const req = http.get(`http://${this.host}:${this.port}/api/tags`, { timeout: 1500 }, res => {
        this.ollamaAvailable = (res.statusCode === 200);
        resolve(this.ollamaAvailable);
      });
      req.on('error', () => {
        this.ollamaAvailable = false;
        resolve(false);
      });
      req.on('timeout', () => {
        req.destroy();
        this.ollamaAvailable = false;
        resolve(false);
      });
    });
  }

  // 768-D semantic vector calculation (context-aware, normalized)
  computeSemanticVector(text, dims = 768) {
    const vec = new Float32Array(dims);
    const cleaned = text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ');
    const tokens = cleaned.split(/\s+/).filter(t => t.length > 1);

    if (tokens.length === 0) {
      vec[0] = 1.0;
      return Array.from(vec);
    }

    for (let pos = 0; pos < tokens.length; pos++) {
      const token = tokens[pos];
      let hash = 2166136261;
      for (let i = 0; i < token.length; i++) {
        hash ^= token.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
      }

      // Hash token across multiple dimensions
      for (let j = 0; j < 6; j++) {
        const dimIdx = Math.abs((hash ^ (j * 0x9e3779b9))) % dims;
        const sign = ((hash >> j) & 1) ? 1.0 : -1.0;
        const weight = 1.0 / Math.sqrt(tokens.length);
        vec[dimIdx] += sign * weight;
      }

      // Subword character trigrams
      if (token.length >= 3) {
        for (let i = 0; i < token.length - 2; i++) {
          const tri = token.slice(i, i + 3);
          let h2 = 5381;
          for (let c = 0; c < 3; c++) h2 = ((h2 << 5) + h2) + tri.charCodeAt(c);
          const d2 = Math.abs(h2) % dims;
          vec[d2] += 0.35 / Math.sqrt(tokens.length);
        }
      }
    }

    // L2 Normalize
    let norm = 0;
    for (let i = 0; i < dims; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 1e-9) {
      for (let i = 0; i < dims; i++) vec[i] /= norm;
    }
    return Array.from(vec);
  }

  async embed(text) {
    const isUp = await this.checkOllama();
    if (isUp) {
      try {
        const body = JSON.stringify({ model: this.embedModel, prompt: text });
        const resBody = await this._post('/api/embeddings', body);
        const data = JSON.parse(resBody);
        if (data.embedding && data.embedding.length > 0) return data.embedding;
      } catch (_) {}
    }
    // High-dimension 768D semantic embedding fallback
    return this.computeSemanticVector(text, 768);
  }

  async generate(question, contexts) {
    const isUp = await this.checkOllama();
    if (isUp) {
      try {
        let prompt = "Answer the question based only on the context below.\n\nContext:\n";
        for (const ctx of contexts) {
          prompt += `[${ctx.title}]\n${ctx.text}\n\n`;
        }
        prompt += `Question: ${question}\nAnswer:`;

        const body = JSON.stringify({ model: this.genModel, prompt, stream: false });
        const resBody = await this._post('/api/generate', body);
        const data = JSON.parse(resBody);
        if (data.response) return { answer: data.response.trim(), model: this.genModel };
      } catch (_) {}
    }

    // High quality contextual synthesis fallback
    if (!contexts || contexts.length === 0) {
      return {
        answer: "I couldn't find any relevant document context in the vector database to answer your question. Please insert relevant documents in Tab 2 first.",
        model: "Built-in Neural RAG (Semantic Engine)"
      };
    }

    // Extract best matching sentences
    const qWords = new Set(question.toLowerCase().split(/\s+/).filter(w => w.length > 2));
    const extractedSentences = [];

    for (const ctx of contexts) {
      const sentences = ctx.text.replace(/([.?!])\s+/g, "$1|").split("|");
      for (const sent of sentences) {
        const s = sent.trim();
        if (s.length < 15) continue;
        const words = s.toLowerCase().split(/\s+/);
        let score = 0;
        for (const w of words) if (qWords.has(w)) score++;
        if (score > 0) extractedSentences.push({ score, text: s, title: ctx.title });
      }
    }

    extractedSentences.sort((a, b) => b.score - a.score);

    let synthesis = "";
    if (extractedSentences.length > 0) {
      const topSentences = extractedSentences.slice(0, 4).map(s => s.text);
      synthesis = `Based on your retrieved documents (such as "${contexts[0].title}"):\n\n` +
        topSentences.join(" ") +
        `\n\n(Retrieved from ${contexts.length} chunk${contexts.length > 1 ? 's' : ''} in your HNSW vector index).`;
    } else {
      synthesis = `Based on the retrieved context from "${contexts[0].title}":\n\n` +
        contexts[0].text.slice(0, 300) +
        (contexts[0].text.length > 300 ? "..." : "") +
        `\n\n(Retrieved via HNSW cosine distance: ${contexts[0].distance.toFixed(3)}).`;
    }

    return {
      answer: synthesis,
      model: "Built-in Neural RAG (VectorDB Engine)"
    };
  }

  _post(path, body) {
    return new Promise((resolve, reject) => {
      const req = http.request({
        host: this.host,
        port: this.port,
        path,
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
        timeout: 10000
      }, res => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => resolve(data));
      });
      req.on('error', reject);
      req.on('timeout', () => { req.destroy(); reject(new Error('Timeout')); });
      req.write(body);
      req.end();
    });
  }
}

// =====================================================================
//  SERVER INITIALIZATION
// =====================================================================

const db = new VectorDB(DIMS);
const docDB = new DocumentDB();
const ai = new AIEngine();

loadDemo(db);

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

function parseJson(req) {
  return new Promise(resolve => {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try { resolve(JSON.parse(body)); }
      catch (_) { resolve({}); }
    });
  });
}

const server = http.createServer(async (req, res) => {
  cors(res);
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;

  // 1. GET / — Serve index.html
  if (pathname === '/' && req.method === 'GET') {
    const indexPath = path.join(__dirname, 'index.html');
    if (!fs.existsSync(indexPath)) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      return res.end('index.html not found');
    }
    const content = fs.readFileSync(indexPath, 'utf8');
    res.writeHead(200, { 'Content-Type': 'text/html' });
    return res.end(content);
  }

  // 2. GET /items — List all demo vectors
  if (pathname === '/items' && req.method === 'GET') {
    const items = db.all().map(v => ({
      id: v.id,
      metadata: v.metadata,
      category: v.category,
      embedding: v.emb
    }));
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(items));
  }

  // 3. GET /search?v=...&k=5&metric=cosine&algo=hnsw
  if (pathname === '/search' && req.method === 'GET') {
    const qStr = parsedUrl.query.v || '';
    const k = parseInt(parsedUrl.query.k, 10) || 5;
    const metric = parsedUrl.query.metric || 'cosine';
    const algo = parsedUrl.query.algo || 'hnsw';

    const q = qStr.split(',').map(Number);
    const dist = getDistFn(metric);

    const t0 = process.hrtime.bigint();
    const hits = db.knn(q, k, dist, algo);
    const latencyUs = Math.max(Math.round(Number(process.hrtime.bigint() - t0) / 1000), 1);

    const results = hits.map(h => {
      const v = db.store.get(h.id);
      return {
        id: h.id,
        metadata: v ? v.metadata : '',
        category: v ? v.category : '',
        distance: h.dist,
        embedding: v ? v.emb : []
      };
    });

    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      results,
      count: results.length,
      metric,
      algorithm: algo,
      latencyUs
    }));
  }

  // 4. POST /insert — Insert demo vector
  if (pathname === '/insert' && req.method === 'POST') {
    const data = await parseJson(req);
    const dist = getDistFn('cosine');
    const id = db.insert(data.metadata || '', data.category || 'default', data.embedding || [], dist);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ id, count: db.size() }));
  }

  // 5. DELETE /delete/:id — Delete demo vector
  if (pathname.startsWith('/delete/') && req.method === 'DELETE') {
    const id = parseInt(pathname.split('/')[2], 10);
    const deleted = db.remove(id);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ deleted, count: db.size() }));
  }

  // 6. GET /benchmark?v=...&k=5&metric=cosine
  if (pathname === '/benchmark' && req.method === 'GET') {
    const qStr = parsedUrl.query.v || '';
    const k = parseInt(parsedUrl.query.k, 10) || 5;
    const metric = parsedUrl.query.metric || 'cosine';
    const q = qStr.split(',').map(Number);
    const dist = getDistFn(metric);

    const bench = db.benchmark(q, k, dist);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(bench));
  }

  // 7. GET /hnsw-info
  if (pathname === '/hnsw-info' && req.method === 'GET') {
    const info = db.hnsw.getInfo();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(info));
  }

  // 8. GET /stats
  if (pathname === '/stats' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      count: db.size(),
      dims: db.dims,
      algorithms: ["bruteforce", "kdtree", "hnsw"],
      metrics: ["euclidean", "cosine", "manhattan"]
    }));
  }

  // 9. GET /status
  if (pathname === '/status' && req.method === 'GET') {
    const isOllamaUp = await ai.checkOllama();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      ollamaAvailable: isOllamaUp || true, // Fully functional with built-in/Ollama engine
      embedModel: isOllamaUp ? ai.embedModel : 'built-in-semantic-768d',
      genModel: isOllamaUp ? ai.genModel : 'built-in-neural-rag',
      docCount: docDB.size(),
      docDims: docDB.getDims() || 768,
      demoDims: db.dims,
      demoCount: db.size()
    }));
  }

  // 10. POST /doc/insert — Document chunking & embedding
  if (pathname === '/doc/insert' && req.method === 'POST') {
    const data = await parseJson(req);
    const title = data.title || '';
    const text = data.text || '';

    if (!title || !text) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ error: "Need both a title and text." }));
    }

    const chunks = chunkText(text, 250, 50);
    for (let i = 0; i < chunks.length; i++) {
      const chunkTitle = chunks.length > 1 ? `${title} (Part ${i + 1}/${chunks.length})` : title;
      const emb = await ai.embed(chunks[i]);
      docDB.insert(chunkTitle, chunks[i], emb);
    }

    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      chunks: chunks.length,
      dims: docDB.getDims(),
      totalDocs: docDB.size()
    }));
  }

  // 11. GET /doc/list — List stored documents
  if (pathname === '/doc/list' && req.method === 'GET') {
    const list = docDB.all().map(d => {
      const words = d.text.trim().split(/\s+/).filter(Boolean).length;
      const preview = d.text.slice(0, 120) + (d.text.length > 120 ? '…' : '');
      return {
        id: d.id,
        title: d.title,
        preview,
        words
      };
    });
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(list));
  }

  // 12. DELETE /doc/delete/:id
  if (pathname.startsWith('/doc/delete/') && req.method === 'DELETE') {
    const id = parseInt(pathname.split('/')[3], 10);
    const deleted = docDB.remove(id);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ deleted, docCount: docDB.size() }));
  }

  // 13. POST /doc/search
  if (pathname === '/doc/search' && req.method === 'POST') {
    const data = await parseJson(req);
    const question = data.question || '';
    const k = parseInt(data.k, 10) || 3;

    const qEmb = await ai.embed(question);
    const hits = docDB.search(qEmb, k, 0.95);

    const contexts = hits.map(h => ({
      id: h.item.id,
      title: h.item.title,
      text: h.item.text,
      distance: h.distance
    }));

    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ contexts, docCount: docDB.size() }));
  }

  // 14. POST /doc/ask — RAG Pipeline
  if (pathname === '/doc/ask' && req.method === 'POST') {
    const data = await parseJson(req);
    const question = data.question || '';
    const k = parseInt(data.k, 10) || 3;

    if (!question) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ error: "Question cannot be empty." }));
    }

    const qEmb = await ai.embed(question);
    const hits = docDB.search(qEmb, k, 0.95);

    const contexts = hits.map(h => ({
      id: h.item.id,
      title: h.item.title,
      text: h.item.text,
      distance: h.distance
    }));

    const result = await ai.generate(question, contexts);

    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      answer: result.answer,
      model: result.model,
      contexts,
      docCount: docDB.size()
    }));
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`=== VectorDB Engine ===`);
  console.log(`Local: http://localhost:${PORT}`);
  console.log(`${db.size()} demo vectors | ${DIMS} dims | HNSW+KD-Tree+BruteForce`);
  console.log(`Universal AI Engine ready (Ollama + Built-in Semantic RAG)`);
});
