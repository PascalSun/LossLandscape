/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * DuckDB database utilities for storing loss landscape data
 */

import duckdb from 'duckdb';
import fs from 'fs';
import path from 'path';

let db: duckdb.Database | null = null;
let dbInitPromise: Promise<duckdb.Database> | null = null;
let schemaInitialized = false;

/**
 * Get the database file path and related file paths
 */
function getDatabasePaths() {
  const dbPath =
    process.env.DUCKDB_PATH ||
    path.join(process.cwd(), 'data', 'app.duckdb');
  const walPath = `${dbPath}.wal`;
  const walShmPath = `${dbPath}.wal.shm`;
  return { dbPath, walPath, walShmPath };
}

/**
 * Attempt to repair corrupted WAL files by removing them
 * This will cause DuckDB to recreate the WAL file on next open
 */
function repairWALFiles(): boolean {
  try {
    const { walPath, walShmPath } = getDatabasePaths();
    let repaired = false;
    
    if (fs.existsSync(walPath)) {
      console.warn(`[DB] Removing corrupted WAL file: ${walPath}`);
      fs.unlinkSync(walPath);
      repaired = true;
    }
    
    if (fs.existsSync(walShmPath)) {
      console.warn(`[DB] Removing corrupted WAL shared memory file: ${walShmPath}`);
      fs.unlinkSync(walShmPath);
      repaired = true;
    }
    
    if (repaired) {
      console.log('[DB] WAL files removed, database will recreate them on next open');
    }
    
    return repaired;
  } catch (e) {
    console.error('[DB] Failed to repair WAL files:', e);
    return false;
  }
}

/**
 * Get or initialize the DuckDB database instance.
 * This ensures the database is fully initialized before use.
 */
async function getDatabaseAsync(): Promise<duckdb.Database> {
  if (db) {
    return db;
  }
  
  if (dbInitPromise) {
    return dbInitPromise;
  }
  
  dbInitPromise = new Promise<duckdb.Database>((resolve, reject) => {
    const { dbPath } = getDatabasePaths();
    console.log(`[DB] Initializing DuckDB at ${dbPath}`);
    
    try {
      fs.mkdirSync(path.dirname(dbPath), { recursive: true });
      const database = new duckdb.Database(dbPath, duckdb.OPEN_READWRITE | duckdb.OPEN_CREATE, (err) => {
        if (err) {
          const errorMessage = err?.message || String(err);
          
          // Check if it's a WAL file corruption error
          const isWALError = 
            errorMessage.includes('WAL') ||
            errorMessage.includes('replaying') ||
            errorMessage.includes('Write-Ahead Log') ||
            errorMessage.includes('GetDefaultDatabase');
          
          if (isWALError) {
            console.error('[DB] WAL file corruption detected, attempting repair...', errorMessage);
            // Try to repair by removing WAL files
            if (repairWALFiles()) {
              // Reset state
              db = null;
              dbInitPromise = null;
              
              // Retry opening the database after WAL repair
              // Use a small delay to ensure files are fully closed
              setTimeout(() => {
                try {
                  const retryDatabase = new duckdb.Database(dbPath, duckdb.OPEN_READWRITE | duckdb.OPEN_CREATE, (retryErr) => {
                    if (retryErr) {
                      console.error('[DB] Failed to open database after WAL repair:', retryErr);
                      db = null;
                      dbInitPromise = null;
                      reject(retryErr);
                    } else {
                      console.log('[DB] Database opened successfully after WAL repair');
                      db = retryDatabase;
                      resolve(retryDatabase);
                    }
                  });
                } catch (retryE) {
                  console.error('[DB] Exception during retry after WAL repair:', retryE);
                  db = null;
                  dbInitPromise = null;
                  reject(retryE);
                }
              }, 200);
              return;
            }
          }
          
          console.error('[DB] Failed to open database:', err);
          db = null;
          dbInitPromise = null;
          reject(err);
        } else {
          console.log('[DB] Database opened successfully');
          db = database;
          resolve(database);
        }
      });
    } catch (e) {
      console.error('[DB] Exception during database initialization:', e);
      dbInitPromise = null;
      reject(e);
    }
  });
  
  return dbInitPromise;
}

/**
 * Get a new connection from the database.
 * Each operation should use a fresh connection to avoid connection state issues.
 */
async function getConnectionAsync(): Promise<duckdb.Connection> {
  const database = await getDatabaseAsync();
  
  try {
    const conn = database.connect();
    
    // Initialize schema on first connection
    if (!schemaInitialized) {
      console.log('[DB] Initializing schema');
      initializeSchema(conn);
      schemaInitialized = true;
    }
    
    return conn;
  } catch (e) {
    console.error('[DB] Failed to create connection:', e);
    // If connection fails, reset database instance to force reconnection
    db = null;
    dbInitPromise = null;
    schemaInitialized = false;
    throw e;
  }
}

/**
 * Execute a database operation with automatic retry on connection errors.
 * This ensures the database is fully initialized and handles connection errors gracefully.
 */
async function withConnectionRetry<T>(
  operation: (conn: duckdb.Connection) => Promise<T>,
  maxRetries: number = 2
): Promise<T> {
  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Get a fresh connection for each operation
      const conn = await getConnectionAsync();
      const result = await operation(conn);
      // Connection will be automatically closed when it goes out of scope
      return result;
    } catch (error: any) {
      lastError = error;
      const errorMessage = error?.message || String(error);
      const errorCode = error?.code || '';
      
      // Check if it's a WAL file corruption error
      const isWALError = 
        errorMessage.includes('WAL') ||
        errorMessage.includes('replaying') ||
        errorMessage.includes('Write-Ahead Log') ||
        errorMessage.includes('GetDefaultDatabase') ||
        errorMessage.includes('Internal Error');
      
      // Check if it's a connection error
      const isConnectionError = 
        errorCode === 'DUCKDB_NODEJS_ERROR' ||
        errorMessage.includes('Connection') ||
        errorMessage.includes('connection') ||
        errorMessage.includes('closed') ||
        errorMessage.includes('never established');
      
      // Handle WAL errors by repairing and retrying
      if (isWALError && attempt < maxRetries) {
        console.warn(`[DB] WAL error detected (attempt ${attempt + 1}/${maxRetries + 1}), attempting repair...`, errorMessage);
        // Reset database instance
        db = null;
        dbInitPromise = null;
        schemaInitialized = false;
        
        // Try to repair WAL files
        if (repairWALFiles()) {
          // Wait a bit before retrying
          await new Promise(resolve => setTimeout(resolve, 200 * (attempt + 1)));
          continue;
        }
      }
      
      if (isConnectionError && attempt < maxRetries) {
        console.warn(`[DB] Connection error (attempt ${attempt + 1}/${maxRetries + 1}), retrying...`, errorMessage);
        // Reset database instance to force reconnection
        db = null;
        dbInitPromise = null;
        schemaInitialized = false;
        // Wait a bit before retrying (exponential backoff)
        await new Promise(resolve => setTimeout(resolve, 100 * Math.pow(2, attempt)));
        continue;
      }
      
      // If not a connection error or max retries reached, throw
      throw error;
    }
  }
  
  // Should never reach here, but TypeScript needs it
  throw lastError || new Error('Unknown database error');
}

function initializeSchema(conn: duckdb.Connection) {
  try {
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
        hessian TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Lightweight migration for existing DB files
    // (DuckDB supports ADD COLUMN IF NOT EXISTS)
    // Wrap each ALTER TABLE in try-catch to handle cases where column already exists
    const alterStatements = [
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS Z TEXT`,
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS export_metadata TEXT`,
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS import_source TEXT`,
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS import_filename TEXT`,
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS imported_at TIMESTAMP`,
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS hessian TEXT`,
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS loss_line_1d TEXT`,
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS X_1d TEXT`,
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS baseline_loss_1d REAL`,
      `ALTER TABLE loss_landscape_data ADD COLUMN IF NOT EXISTS config_yml TEXT`,
    ];
    
    for (const sql of alterStatements) {
      try {
        conn.run(sql);
      } catch (e: any) {
        // Ignore errors for columns that might already exist
        // DuckDB's IF NOT EXISTS should handle this, but just in case
        if (!e.message?.includes('already exists')) {
          console.warn(`[DB] Schema migration warning:`, e.message);
        }
      }
    }
    
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
  } catch (e: any) {
    console.error('[DB] Failed to initialize schema:', e);
    // Reset schema initialization flag so it can be retried
    schemaInitialized = false;
    throw e;
  }
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
  loss_line_1d?: number[]; // 1D loss line for 1D visualization
  X_1d?: number[]; // 1D X axis for 1D visualization
  baseline_loss_1d?: number; // Baseline loss for 1D visualization
  trajectory_data?: {
    traj_1?: number[];
    traj_2?: number[];
    traj_3?: number[];
    epochs?: number[];
    losses?: number[];
    val_losses?: number[];
  };
  export_metadata?: any; // Metadata from JSON file (training metadata) or .export.meta.json file
  hessian?: any;
  config_yml?: string; // Content of config.yml file if found in run directory
  import_source?: string; // Source of import: 'upload', 'scan', 'run', 'generate', etc.
  import_filename?: string; // Original filename if imported
  imported_at?: Date | string; // When the data was imported
  skipDuplicateCheck?: boolean; // If true, always insert new record (for generate API)
}

export async function saveLossLandscape(data: LossLandscapeData): Promise<number> {
  return withConnectionRetry(async (conn) => {
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
    // We include import_filename in the duplicate check to distinguish between different files
    // from the same run/config, as they may have different data even if mode/grid_size are the same.
    if (!data.skipDuplicateCheck && (data.config_path || data.run_dir)) {
      const existingId = await new Promise<number | null>((resolve, reject) => {
        const sql = `
          SELECT id FROM loss_landscape_data 
          WHERE 
            (config_path = ? OR (config_path IS NULL AND ? IS NULL)) AND
            (run_dir = ? OR (run_dir IS NULL AND ? IS NULL)) AND
            mode = ? AND
            grid_size = ? AND
            (import_filename = ? OR (import_filename IS NULL AND ? IS NULL))
          LIMIT 1
        `;
        conn.all(sql, 
          data.config_path || null, data.config_path || null,
          data.run_dir || null, data.run_dir || null,
          data.mode,
          data.grid_size,
          data.import_filename || null, data.import_filename || null
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

    return new Promise<number>((resolve, reject) => {
      const sql = `
        INSERT INTO loss_landscape_data
          (id, config_path, run_dir, mode, direction, grid_size, X, Y, Z, loss_grid_2d, loss_grid_3d, baseline_loss, loss_line_1d, X_1d, baseline_loss_1d, trajectory_data, hessian, export_metadata, config_yml, import_source, import_filename, imported_at)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        data.loss_line_1d ? JSON.stringify(data.loss_line_1d) : null,
        data.X_1d ? JSON.stringify(data.X_1d) : null,
        data.baseline_loss_1d || null,
        data.trajectory_data ? JSON.stringify(data.trajectory_data) : null,
        data.hessian ? JSON.stringify(data.hessian) : null,
        data.export_metadata ? JSON.stringify(data.export_metadata) : null,
        data.config_yml || null,
        data.import_source || null,
        data.import_filename || null,
        data.imported_at ? (typeof data.imported_at === 'string' ? data.imported_at : data.imported_at.toISOString()) : new Date().toISOString(),
      ];
      conn.all(sql, ...params, (err: Error | null, rows: any[]) => {
        if (err) return reject(err);
        resolve(Number(rows?.[0]?.id ?? nextId));
      });
    });
  });
}

export async function getLossLandscape(id: number): Promise<LossLandscapeData | null> {
  return withConnectionRetry(async (conn) => {
    return new Promise<LossLandscapeData | null>((resolve, reject) => {
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
              loss_line_1d: row.loss_line_1d ? JSON.parse(row.loss_line_1d) : undefined,
              X_1d: row.X_1d ? JSON.parse(row.X_1d) : undefined,
              baseline_loss_1d: row.baseline_loss_1d || undefined,
              trajectory_data: row.trajectory_data ? JSON.parse(row.trajectory_data) : undefined,
            hessian: row.hessian ? JSON.parse(row.hessian) : undefined,
            export_metadata: row.export_metadata ? JSON.parse(row.export_metadata) : undefined,
            config_yml: row.config_yml || undefined,
            import_source: row.import_source,
            import_filename: row.import_filename,
            imported_at: row.imported_at,
          });
          }
        }
      );
    });
  });
}

export async function listLossLandscapes(): Promise<LossLandscapeData[]> {
  return withConnectionRetry(async (conn) => {
    return new Promise<LossLandscapeData[]>((resolve, reject) => {
      // For history list, we only need metadata fields, not the full data arrays
      // This improves performance and reduces parsing errors
      conn.all(`SELECT id, config_path, run_dir, mode, direction, grid_size, baseline_loss, 
                trajectory_data, hessian, export_metadata, config_yml, import_source, 
                import_filename, imported_at, created_at,
                X, Y, Z, loss_grid_2d, loss_grid_3d, loss_line_1d, X_1d, baseline_loss_1d
                FROM loss_landscape_data ORDER BY created_at DESC`, 
        (err: Error | null, rows: any[]) => {
        if (err) {
          reject(err);
        } else {
          resolve(
            rows.map((row) => {
              try {
                return {
                  id: typeof row.id === 'bigint' ? Number(row.id) : Number(row.id),
                  config_path: row.config_path,
                  run_dir: row.run_dir,
                  mode: row.mode,
                  direction: row.direction,
                  grid_size: row.grid_size,
                  // Parse large data fields with error handling
                  X: row.X ? (() => { try { return JSON.parse(row.X); } catch { return []; } })() : [],
                  Y: row.Y ? (() => { try { return JSON.parse(row.Y); } catch { return []; } })() : [],
                  Z: row.Z ? (() => { try { return JSON.parse(row.Z); } catch { return undefined; } })() : undefined,
                  loss_grid_2d: row.loss_grid_2d ? (() => { try { return JSON.parse(row.loss_grid_2d); } catch { return []; } })() : [],
                  loss_grid_3d: row.loss_grid_3d ? (() => { try { return JSON.parse(row.loss_grid_3d); } catch { return undefined; } })() : undefined,
                  baseline_loss: row.baseline_loss,
                  loss_line_1d: row.loss_line_1d ? (() => { try { return JSON.parse(row.loss_line_1d); } catch { return undefined; } })() : undefined,
                  X_1d: row.X_1d ? (() => { try { return JSON.parse(row.X_1d); } catch { return undefined; } })() : undefined,
                  baseline_loss_1d: row.baseline_loss_1d || undefined,
                  trajectory_data: row.trajectory_data ? (() => { try { return JSON.parse(row.trajectory_data); } catch { return undefined; } })() : undefined,
                  hessian: row.hessian ? (() => { try { return JSON.parse(row.hessian); } catch { return undefined; } })() : undefined,
                  export_metadata: row.export_metadata ? (() => { try { return JSON.parse(row.export_metadata); } catch { return undefined; } })() : undefined,
                  config_yml: row.config_yml || undefined,
                  import_source: row.import_source,
                  import_filename: row.import_filename,
                  imported_at: row.imported_at,
                };
              } catch (e: any) {
                console.error(`[listLossLandscapes] Error parsing row id=${row.id}:`, e);
                // Return minimal data if parsing fails
                return {
                  id: typeof row.id === 'bigint' ? Number(row.id) : Number(row.id),
                  config_path: row.config_path,
                  run_dir: row.run_dir,
                  mode: row.mode || '2d',
                  direction: row.direction || 'unknown',
                  grid_size: row.grid_size || 0,
                  X: [],
                  Y: [],
                  loss_grid_2d: [],
                  baseline_loss: row.baseline_loss || 0,
                  import_source: row.import_source,
                  import_filename: row.import_filename,
                  imported_at: row.imported_at,
                };
              }
            })
          );
        }
      });
    });
  });
}

/**
 * Execute a database update operation with automatic retry on connection errors
 * This is a fire-and-forget operation that doesn't return a value
 */
export async function executeUpdate(
  sql: string,
  ...params: any[]
): Promise<void> {
  return withConnectionRetry(async (conn) => {
    return new Promise<void>((resolve, reject) => {
      conn.run(sql, ...params, (err: Error | null) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });
  });
}

/**
 * Execute a database delete operation with automatic retry on connection errors
 */
export async function executeDelete(
  sql: string,
  ...params: any[]
): Promise<void> {
  return withConnectionRetry(async (conn) => {
    return new Promise<void>((resolve, reject) => {
      conn.run(sql, ...params, (err: Error | null) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });
  });
}

