/**
 * Learner progress storage — issue: returning learners.
 *
 * Persists SRS cards per (language, phoneme) in IndexedDB (with a
 * localStorage fallback where IDB is unavailable, e.g. private-mode
 * Safari). Everything is on-device; no account, no server — consistent
 * with the app's privacy stance (docs/TDS.md "Privacy, Offline & Performance").
 *
 * Layout ("bpt-progress" DB):
 *   store "cards"    key "<lang>:<phoneme>"  → SrsCard
 *   store "meta"     key "stats"             → { attempts, sessions, firstVisitAt }
 */
import type { SrsCard } from "./srs.ts";

const DB_NAME = "bpt-progress";
const DB_VERSION = 1;
const LS_PREFIX = "bpt-card:";

export interface ProgressStats {
  attempts: number;
  sessions: number;
  firstVisitAt: number;
  lastVisitAt: number;
}

const DEFAULT_STATS: ProgressStats = { attempts: 0, sessions: 0, firstVisitAt: 0, lastVisitAt: 0 };

function cardKey(lang: string, phoneme: string): string {
  return `${lang}:${phoneme}`;
}

/** Storage abstraction so SRS logic and UI don't care where data lives. */
export interface ProgressStore {
  getCard(lang: string, phoneme: string): Promise<SrsCard | undefined>;
  putCard(card: SrsCard): Promise<void>;
  getAllCards(lang: string): Promise<SrsCard[]>;
  getStats(): Promise<ProgressStats>;
  recordSession(attemptCount: number): Promise<ProgressStats>;
}

/** IndexedDB-backed store. */
 class IdbStore implements ProgressStore {
  private db(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains("cards")) db.createObjectStore("cards");
        if (!db.objectStoreNames.contains("meta")) db.createObjectStore("meta");
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  private async tx<T>(store: string, mode: IDBTransactionMode, run: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const db = await this.db();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(store, mode);
      const req = run(tx.objectStore(store));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
      tx.oncomplete = () => db.close();
    });
  }

  getCard(lang: string, phoneme: string): Promise<SrsCard | undefined> {
    return this.tx("cards", "readonly", (s) => s.get(cardKey(lang, phoneme)));
  }

  async putCard(card: SrsCard): Promise<void> {
    await this.tx("cards", "readwrite", (s) => s.put(card, cardKey(card.lang, card.phoneme)));
  }

  async getAllCards(lang: string): Promise<SrsCard[]> {
    const all = await this.tx<SrsCard[]>("cards", "readonly", (s) => s.getAll() as IDBRequest<SrsCard[]>);
    return all.filter((c) => c.lang === lang);
  }

  getStats(): Promise<ProgressStats> {
    return this.tx("meta", "readonly", (s) => s.get("stats")).then((v) => v ?? { ...DEFAULT_STATS });
  }

  async recordSession(attemptCount: number): Promise<ProgressStats> {
    const prev = await this.getStats();
    const now = Date.now();
    const stats: ProgressStats = {
      attempts: prev.attempts + attemptCount,
      sessions: prev.sessions + 1,
      firstVisitAt: prev.firstVisitAt || now,
      lastVisitAt: now,
    };
    await this.tx("meta", "readwrite", (s) => s.put(stats, "stats"));
    return stats;
  }
}

/** localStorage fallback (Safari private mode, IDB failures). */
class LocalStore implements ProgressStore {
  async getCard(lang: string, phoneme: string): Promise<SrsCard | undefined> {
    const raw = localStorage.getItem(LS_PREFIX + cardKey(lang, phoneme));
    return raw ? (JSON.parse(raw) as SrsCard) : undefined;
  }

  async putCard(card: SrsCard): Promise<void> {
    localStorage.setItem(LS_PREFIX + cardKey(card.lang, card.phoneme), JSON.stringify(card));
  }

  async getAllCards(lang: string): Promise<SrsCard[]> {
    const out: SrsCard[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)!;
      if (!key.startsWith(LS_PREFIX)) continue;
      const card = JSON.parse(localStorage.getItem(key)!) as SrsCard;
      if (card.lang === lang) out.push(card);
    }
    return out;
  }

  async getStats(): Promise<ProgressStats> {
    const raw = localStorage.getItem("bpt-stats");
    return raw ? (JSON.parse(raw) as ProgressStats) : { ...DEFAULT_STATS };
  }

  async recordSession(attemptCount: number): Promise<ProgressStats> {
    const prev = await this.getStats();
    const now = Date.now();
    const stats: ProgressStats = {
      attempts: prev.attempts + attemptCount,
      sessions: prev.sessions + 1,
      firstVisitAt: prev.firstVisitAt || now,
      lastVisitAt: now,
    };
    localStorage.setItem("bpt-stats", JSON.stringify(stats));
    return stats;
  }
}

/** In-memory store (SSR/tests) — nothing persists. */
export class MemoryStore implements ProgressStore {
  private cards = new Map<string, SrsCard>();
  private stats: ProgressStats = { ...DEFAULT_STATS };

  async getCard(lang: string, phoneme: string): Promise<SrsCard | undefined> {
    return this.cards.get(cardKey(lang, phoneme));
  }
  async putCard(card: SrsCard): Promise<void> {
    this.cards.set(cardKey(card.lang, card.phoneme), card);
  }
  async getAllCards(lang: string): Promise<SrsCard[]> {
    return [...this.cards.values()].filter((c) => c.lang === lang);
  }
  async getStats(): Promise<ProgressStats> {
    return { ...this.stats };
  }
  async recordSession(attemptCount: number): Promise<ProgressStats> {
    this.stats = {
      attempts: this.stats.attempts + attemptCount,
      sessions: this.stats.sessions + 1,
      firstVisitAt: this.stats.firstVisitAt || Date.now(),
      lastVisitAt: Date.now(),
    };
    return { ...this.stats };
  }
}

/** Best available store for this browser (async so IDB can fail safely). */
export async function openProgressStore(): Promise<ProgressStore> {
  if (typeof indexedDB !== "undefined") {
    try {
      // probe IDB actually works (private mode throws on open in old Safari)
      const probe = new Promise<boolean>((resolve) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onsuccess = () => {
          req.result.close();
          resolve(true);
        };
        req.onerror = () => resolve(false);
        // don't hang forever
        setTimeout(() => resolve(false), 2000);
      });
      if (await probe) return new IdbStore();
    } catch {
      /* fall through */
    }
  }
  if (typeof localStorage !== "undefined") {
    try {
      localStorage.setItem("bpt-probe", "1");
      localStorage.removeItem("bpt-probe");
      return new LocalStore();
    } catch {
      /* fall through */
    }
  }
  return new MemoryStore();
}
