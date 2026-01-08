/**
 * DuckDB database utilities for storing loss landscape data
 */

import duckdb from 'duckdb';
import fs from 'fs';
import path from 'path';

let db: duckdb.Database | null = null;
let schemaInitialized = false;

export function getDatabase(): duckdb.Database {
  if (!db) {
    const dbPath =
      process.env.DUCKDB_PATH ||
      path.join(process.cwd(), 'data', 'app.duckdb');
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    db = new duckdb.Database(dbPath);
  }
  return db;
}

export function getConnection(): duckdb.Connection {
  const conn = getDatabase().connect();
  if (!schemaInitialized) {
    initializeSchema(conn);
    schemaInitialized = true;
  }
  return conn;
}

function initializeSchema(conn: duckdb.Connection) {
  
  // Create loss_landscape_data table
  conn.run(`
    CREATE TABLE IF NOT EXISTS loss_landscape_data (
      id BIGINT PRIMARY KEY,
      config_path TEXT,
      run_dir TEXT,
      mode TEXT,
      direction TEXT,
      grid_size INTEGER,
      X TEXT,
      Y TEXT,
      Z TEXT,
      loss_grid_2d TEXT,
      loss_grid_3d TEXT,
      baseline_loss REAL,
      trajectory_data TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // Lightweight migration for existing DB files
  // (DuckDB supports ADD COLUMN IF NOT EXISTS)
  conn.run(`ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS Z TEXT`);
  conn.run(`ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS export_metadata TEXT`);
  conn.run(`ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS import_source TEXT`);
  conn.run(`ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS import_filename TEXT`);
  conn.run(`ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS imported_at TIMESTAMP`);
  
  // Create trajectory_points table
  conn.run(`
    CREATE TABLE IF NOT EXISTS trajectory_points (
      id INTEGER PRIMARY KEY,
      landscape_id INTEGER,
      epoch INTEGER,
      x REAL,
      y REAL,
      z REAL,
      loss REAL,
      FOREIGN KEY (landscape_id) REFERENCES loss_landscape_data(id)
    )
  `);
}

export interface LossLandscapeData {
  id?: number;
  config_path?: string;
  run_dir?: string;
  mode: string;
  direction: string;
  grid_size: number;
  X: number[][];
  Y: number[][];
  Z?: number[];
  loss_grid_2d: number[][];
  loss_grid_3d?: number[][][];
  baseline_loss: number;
  trajectory_data?: {
    traj_1: number[];
    traj_2: number[];
    traj_3?: number[];
    epochs: number[];
    losses?: number[];
    val_losses?: number[];
  };
  export_metadata?: any; // Metadata from JSON file (training metadata) or .export.meta.json file
  import_source?: string; // Source of import: 'upload', 'scan', 'run', 'generate', etc.
  import_filename?: string; // Original filename if imported
  imported_at?: Date | string; // When the data was imported
  skipDuplicateCheck?: boolean; // If true, always insert new record (for generate API)
}

export async function saveLossLandscape(data: LossLandscapeData): Promise<number> {
  const conn = getConnection();

  // Use a DB-derived incremental id to stay compatible with older DB files
  // that may have id stored as INT32.
    const nextId = await new Promise<number>((resolve, reject) => {
    conn.all(
      `SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM loss_landscape_data`,
      (err: Error | null, rows: any[]) => {
        if (err) return reject(err);
        resolve(Number(rows?.[0]?.next_id ?? 1));
      }
    );
  });
  
  // Check for duplicates only for NPZ imports (not for generate API)
  // Generate API should always create new records, even with same parameters,
  // because a config/run can generate multiple landscapes with different parameters
  // or the same parameters at different times.
  if (!data.skipDuplicateCheck && (data.config_path || data.run_dir)) {
      const existingId = await new Promise<number | null>((resolve, reject) => {
        const sql = `
          SELECT id FROM loss_landscape_data 
          WHERE 
            (config_path = ? OR (config_path IS NULL AND ? IS NULL)) AND
            (run_dir = ? OR (run_dir IS NULL AND ? IS NULL)) AND
            mode = ? AND
            grid_size = ?
          LIMIT 1
        `;
        conn.all(sql, 
          data.config_path || null, data.config_path || null,
          data.run_dir || null, data.run_dir || null,
          data.mode,
          data.grid_size
        , (err: Error | null, rows: any[]) => {
            if (err) return reject(err);
            if (rows.length > 0) resolve(Number(rows[0].id));
            else resolve(null);
        });
      });
      
      if (existingId !== null) {
          console.log(`[saveLossLandscape] Found duplicate entry (id=${existingId}), skipping insert.`);
          return existingId;
      }
  }

  return new Promise((resolve, reject) => {
    const sql = `
      INSERT INTO loss_landscape_data
        (id, config_path, run_dir, mode, direction, grid_size, X, Y, Z, loss_grid_2d, loss_grid_3d, baseline_loss, trajectory_data, export_metadata, import_source, import_filename, imported_at)
      VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      RETURNING id
    `;
    const params = [
      nextId,
        data.config_path || null,
        data.run_dir || null,
        data.mode,
        data.direction,
        data.grid_size,
        JSON.stringify(data.X),
        JSON.stringify(data.Y),
        data.Z ? JSON.stringify(data.Z) : null,
        JSON.stringify(data.loss_grid_2d),
        data.loss_grid_3d ? JSON.stringify(data.loss_grid_3d) : null,
        data.baseline_loss,
        data.trajectory_data ? JSON.stringify(data.trajectory_data) : null,
        data.export_metadata ? JSON.stringify(data.export_metadata) : null,
        data.import_source || null,
        data.import_filename || null,
        data.imported_at ? (typeof data.imported_at === 'string' ? data.imported_at : data.imported_at.toISOString()) : new Date().toISOString(),
    ];
    conn.all(sql, ...params, (err: Error | null, rows: any[]) => {
      if (err) return reject(err);
      resolve(Number(rows?.[0]?.id ?? nextId));
    });
  });
}

export async function getLossLandscape(id: number): Promise<LossLandscapeData | null> {
  const conn = getConnection();
  
  return new Promise((resolve, reject) => {
    conn.all(
      `SELECT * FROM loss_landscape_data WHERE id = ?`,
      id,
      (err: Error | null, rows: any[]) => {
        if (err) {
          reject(err);
        } else if (rows.length === 0) {
          resolve(null);
        } else {
          const row = rows[0];
          resolve({
            id: typeof row.id === 'bigint' ? Number(row.id) : Number(row.id),
            config_path: row.config_path,
            run_dir: row.run_dir,
            mode: row.mode,
            direction: row.direction,
            grid_size: row.grid_size,
            X: JSON.parse(row.X),
            Y: JSON.parse(row.Y),
            Z: row.Z ? JSON.parse(row.Z) : undefined,
            loss_grid_2d: JSON.parse(row.loss_grid_2d),
            loss_grid_3d: row.loss_grid_3d ? JSON.parse(row.loss_grid_3d) : undefined,
            baseline_loss: row.baseline_loss,
            trajectory_data: row.trajectory_data ? JSON.parse(row.trajectory_data) : undefined,
            export_metadata: row.export_metadata ? JSON.parse(row.export_metadata) : undefined,
            import_source: row.import_source,
            import_filename: row.import_filename,
            imported_at: row.imported_at,
          });
        }
      }
    );
  });
}

export async function listLossLandscapes(): Promise<LossLandscapeData[]> {
  const conn = getConnection();
  
  return new Promise((resolve, reject) => {
    conn.all(`SELECT * FROM loss_landscape_data ORDER BY created_at DESC`, (err: Error | null, rows: any[]) => {
        if (err) {
          reject(err);
        } else {
          resolve(
            rows.map((row) => ({
              id: typeof row.id === 'bigint' ? Number(row.id) : Number(row.id),
              config_path: row.config_path,
              run_dir: row.run_dir,
              mode: row.mode,
              direction: row.direction,
              grid_size: row.grid_size,
              X: JSON.parse(row.X),
              Y: JSON.parse(row.Y),
              Z: row.Z ? JSON.parse(row.Z) : undefined,
              loss_grid_2d: JSON.parse(row.loss_grid_2d),
              loss_grid_3d: row.loss_grid_3d ? JSON.parse(row.loss_grid_3d) : undefined,
              baseline_loss: row.baseline_loss,
              trajectory_data: row.trajectory_data ? JSON.parse(row.trajectory_data) : undefined,
              export_metadata: row.export_metadata ? JSON.parse(row.export_metadata) : undefined,
              import_source: row.import_source,
              import_filename: row.import_filename,
              imported_at: row.imported_at,
            }))
          );
        }
      });
  });
}

