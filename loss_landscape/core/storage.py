"""
LandscapeStorage - DuckDB数据持久化模块
"""

import duckdb
import os
from typing import Dict, Any, Optional, List
import numpy as np


class LandscapeStorage:
    """
    使用DuckDB存储Loss Landscape数据。
    
    数据格式：
    - 表: landscape_points
    - 字段: epoch (int), x (float), y (float), loss (float), is_trajectory (bool)
    """
    
    def __init__(self, db_path: str, mode: str = 'create'):
        """
        初始化存储实例。
        
        Args:
            db_path: DuckDB文件路径（.duckdb或.landscape）
            mode: 'create' 创建新文件，'append' 追加到现有文件
        """
        self.db_path = db_path
        self.mode = mode
        
        # 确保目录存在
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:  # 如果路径包含目录
            os.makedirs(db_dir, exist_ok=True)
        
        # 连接数据库
        if mode == 'create' and os.path.exists(db_path):
            os.remove(db_path)  # 创建模式：删除旧文件
        
        self.conn = duckdb.connect(db_path)
        self._initialize_schema()
        
    def _initialize_schema(self):
        """初始化数据库表结构"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS landscape_points (
                epoch INTEGER,
                x REAL,
                y REAL,
                z REAL,
                loss REAL,
                is_trajectory BOOLEAN
            )
        """)
        
        # 创建metadata表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # 创建索引以提高查询性能
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_epoch ON landscape_points(epoch)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trajectory ON landscape_points(is_trajectory)
        """)
    
    def save_surface(self, surface_data: Dict[str, Any]):
        """
        保存表面数据到DuckDB。
        
        Args:
            surface_data: build_surface()返回的字典
        """
        try:
            X = np.array(surface_data['X'])
            Y = np.array(surface_data['Y'])
            loss_grid = np.array(surface_data['loss_grid_2d'])
            
            grid_size = len(X)
            epoch = surface_data.get('epoch', 0)  # 表面数据默认epoch=0
            
            # 准备批量插入数据
            data = []
            for i in range(grid_size):
                for j in range(grid_size):
                    data.append(
                        [
                            epoch,
                            float(X[i, j]),
                            float(Y[i, j]),
                            None,  # z坐标（3D体积使用）
                            float(loss_grid[i, j]),
                            False,  # is_trajectory
                        ]
                    )
            
            # 批量插入
            if data:
                self.conn.executemany(
                    "INSERT INTO landscape_points (epoch, x, y, z, loss, is_trajectory) VALUES (?, ?, ?, ?, ?, ?)",
                    data
                )
                self.conn.commit()
                print(f"[LandscapeStorage] Saved {len(data)} 2D surface points to database")
            else:
                print(f"[LandscapeStorage] WARNING: No data to save for surface (grid_size={grid_size})")
        except Exception as e:
            print(f"[LandscapeStorage] ERROR saving surface data: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Persist true baseline loss separately.
        # NOTE: baseline_loss might not be present in the stored grid when grid_size is even (0 not sampled),
        # so exporting baseline as min(loss_grid) is incorrect and can be wildly off.
        if "baseline_loss" in surface_data:
            try:
                baseline = float(surface_data["baseline_loss"])
                self.conn.execute("DELETE FROM metadata WHERE key = 'baseline_loss_2d'")
                self.conn.execute(
                    "INSERT INTO metadata (key, value) VALUES ('baseline_loss_2d', ?)",
                    [str(baseline)],
                )
                self.conn.commit()
            except Exception:
                pass

    def save_volume(self, volume_data: Dict[str, Any]):
        """
        保存3D体积数据到DuckDB。
        
        Args:
            volume_data: Explorer.build_volume() 返回的字典
        """
        X = np.array(volume_data["X"])
        Y = np.array(volume_data["Y"])
        Z = np.array(volume_data["Z"])
        loss_grid = np.array(volume_data["loss_grid_3d"])

        if X.shape != Y.shape or X.shape != Z.shape or X.shape != loss_grid.shape:
            raise ValueError("X, Y, Z, loss_grid_3d 形状必须一致")

        grid_size = X.shape[0]
        epoch = volume_data.get("epoch", 0)

        data = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    data.append(
                        [
                            epoch,
                            float(X[i, j, k]),
                            float(Y[i, j, k]),
                            float(Z[i, j, k]),
                            float(loss_grid[i, j, k]),
                            False,  # is_trajectory
                        ]
                    )

        if data:
            self.conn.executemany(
                "INSERT INTO landscape_points (epoch, x, y, z, loss, is_trajectory) VALUES (?, ?, ?, ?, ?, ?)",
                data,
            )
            self.conn.commit()

        if "baseline_loss" in volume_data:
            try:
                baseline = float(volume_data["baseline_loss"])
                self.conn.execute("DELETE FROM metadata WHERE key = 'baseline_loss_3d'")
                self.conn.execute(
                    "INSERT INTO metadata (key, value) VALUES ('baseline_loss_3d', ?)",
                    [str(baseline)],
                )
                self.conn.commit()
            except Exception:
                pass

    def save_line(self, line_data: Dict[str, Any]):
        """
        保存1D线条数据到DuckDB。
        
        Args:
            line_data: Explorer.build_line() 返回的字典
        """
        try:
            X = np.array(line_data['X'])
            loss_line = np.array(line_data['loss_line_1d'])
            
            grid_size = len(X)
            epoch = line_data.get('epoch', 0)  # 线条数据默认epoch=0
            
            # 准备批量插入数据
            data = []
            for i in range(grid_size):
                data.append(
                    [
                        epoch,
                        float(X[i]),
                        None,  # y坐标（1D线条不使用）
                        None,  # z坐标（1D线条不使用）
                        float(loss_line[i]),
                        False,  # is_trajectory
                    ]
                )
            
            # 批量插入
            if data:
                self.conn.executemany(
                    "INSERT INTO landscape_points (epoch, x, y, z, loss, is_trajectory) VALUES (?, ?, ?, ?, ?, ?)",
                    data
                )
                self.conn.commit()
                print(f"[LandscapeStorage] Saved {len(data)} 1D line points to database")
            else:
                print(f"[LandscapeStorage] WARNING: No data to save for line (grid_size={grid_size})")
        except Exception as e:
            print(f"[LandscapeStorage] ERROR saving line data: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Persist true baseline loss separately.
        if "baseline_loss" in line_data:
            try:
                baseline = float(line_data["baseline_loss"])
                self.conn.execute("DELETE FROM metadata WHERE key = 'baseline_loss_1d'")
                self.conn.execute(
                    "INSERT INTO metadata (key, value) VALUES ('baseline_loss_1d', ?)",
                    [str(baseline)],
                )
                self.conn.commit()
            except Exception:
                pass
    
    def save_trajectory(self, trajectory_data: Dict[str, Any]):
        """
        保存轨迹数据到DuckDB。
        
        Args:
            trajectory_data: build_trajectory()返回的字典
        """
        traj_1 = trajectory_data['traj_1']
        traj_2 = trajectory_data['traj_2']
        traj_3 = trajectory_data.get('traj_3')
        epochs = trajectory_data['epochs']
        
        # 计算loss（需要重新评估，这里先设为None，后续可以优化）
        # 为了简化，我们假设loss已经在其他地方计算
        losses = trajectory_data.get('losses', [None] * len(traj_1))
        val_losses = trajectory_data.get('val_losses', [None] * len(traj_1))
        
        # 准备批量插入数据
        # 注意：当前schema只支持一个loss字段，我们保存train loss
        # val_losses会通过export_for_frontend导出
        data = []
        for i, epoch in enumerate(epochs):
            data.append([
                int(epoch),
                float(traj_1[i]),
                float(traj_2[i]),
                float(traj_3[i]) if traj_3 is not None and i < len(traj_3) else None,  # z坐标
                float(losses[i]) if losses[i] is not None else None,
                True,  # is_trajectory
            ])
        
        # 批量插入
        if data:
            self.conn.executemany(
                "INSERT INTO landscape_points (epoch, x, y, z, loss, is_trajectory) VALUES (?, ?, ?, ?, ?, ?)",
                data
            )
            self.conn.commit()
        
        # 保存val_losses到metadata（如果存在）
        if any(vl is not None for vl in val_losses):
            import json
            val_losses_json = json.dumps(val_losses, indent=2)
            self.conn.execute("""
                DELETE FROM metadata WHERE key = 'trajectory_val_losses'
            """)
            self.conn.execute("""
                INSERT INTO metadata (key, value) 
                VALUES ('trajectory_val_losses', ?)
            """, [val_losses_json])
            self.conn.commit()
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """
        保存metadata信息到数据库。
        
        Args:
            metadata: 包含各种metadata信息的字典，会被序列化为JSON存储
        """
        import json
        
        # 将整个metadata字典序列化为JSON
        metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
        
        # 先删除旧记录（如果存在），然后插入新记录
        self.conn.execute("""
            DELETE FROM metadata WHERE key = 'full_metadata'
        """)
        self.conn.execute("""
            INSERT INTO metadata (key, value) 
            VALUES ('full_metadata', ?)
        """, [metadata_json])
        self.conn.commit()
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        从数据库读取metadata信息。
        
        Returns:
            metadata字典，如果不存在则返回None
        """
        try:
            result = self.conn.execute("""
                SELECT value FROM metadata WHERE key = 'full_metadata'
            """).fetchone()
            
            if result:
                import json
                return json.loads(result[0])
            return None
        except Exception:
            return None
    
    def close(self):
        """关闭连接并刷新数据"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def export_for_frontend(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        导出数据为前端可用的格式（兼容现有前端的数据结构）。
        
        Args:
            output_path: 可选，如果提供则保存为JSON文件
            
        Returns:
            包含X, Y, loss_grid_2d等字段的字典
        """
        # 查询1D线条数据（y和z都为空）
        line_df = self.conn.execute(
            """
            SELECT x, loss, epoch
            FROM landscape_points
            WHERE is_trajectory = FALSE AND y IS NULL AND z IS NULL
            ORDER BY epoch, x
        """
        ).df()
        
        # 查询二维表面数据（z为空，但y不为空）
        surface_df = self.conn.execute(
            """
            SELECT x, y, loss, epoch
            FROM landscape_points
            WHERE is_trajectory = FALSE AND z IS NULL AND y IS NOT NULL
            ORDER BY epoch, x, y
        """
        ).df()
        
        # 查询轨迹数据
        trajectory_df = self.conn.execute(
            """
            SELECT epoch, x, y, z, loss
            FROM landscape_points
            WHERE is_trajectory = TRUE
            ORDER BY epoch
        """
        ).df()
        
        # 初始化result字典
        result = {}
        
        # 先构建2D网格数据（优先，因为这是主要视图）
        if len(surface_df) > 0:
            # Prefer the true baseline loss saved by Explorer, if available.
            baseline_loss_2d = None
            try:
                row = self.conn.execute(
                    "SELECT value FROM metadata WHERE key = 'baseline_loss_2d'"
                ).fetchone()
                if row and row[0] is not None:
                    baseline_loss_2d = float(row[0])
            except Exception:
                baseline_loss_2d = None

            # 获取唯一的x和y值
            unique_x = sorted(surface_df['x'].unique())
            unique_y = sorted(surface_df['y'].unique())
            
            grid_size = len(unique_x)
            
            # 构建网格
            X = np.zeros((grid_size, grid_size))
            Y = np.zeros((grid_size, grid_size))
            loss_grid = np.zeros((grid_size, grid_size))
            
            for idx, row in surface_df.iterrows():
                i = unique_x.index(row['x'])
                j = unique_y.index(row['y'])
                X[i, j] = row['x']
                Y[i, j] = row['y']
                loss_grid[i, j] = row['loss']
            
            # 2D数据总是使用X和Y字段（主要视图）
            result["X"] = X.tolist()
            result["Y"] = Y.tolist()
            result["loss_grid_2d"] = loss_grid.tolist()
            result["baseline_loss"] = float(baseline_loss_2d) if baseline_loss_2d is not None else float(loss_grid.min())
            result["grid_size"] = grid_size
            result["mode"] = "2d"

        # 然后构建1D线条数据（作为附加数据）
        if len(line_df) > 0:
            # Prefer the true baseline loss saved by Explorer, if available.
            baseline_loss_1d = None
            try:
                row = self.conn.execute(
                    "SELECT value FROM metadata WHERE key = 'baseline_loss_1d'"
                ).fetchone()
                if row and row[0] is not None:
                    baseline_loss_1d = float(row[0])
            except Exception:
                baseline_loss_1d = None

            # 获取唯一的x值并排序
            unique_x = sorted(line_df['x'].unique())
            grid_size = len(unique_x)
            
            # 构建1D数组
            X_1d = np.zeros(grid_size)
            loss_line = np.zeros(grid_size)
            
            for idx, row in line_df.iterrows():
                i = unique_x.index(row['x'])
                X_1d[i] = row['x']
                loss_line[i] = row['loss']
            
            # 1D数据使用独立的字段名
            result["X_1d"] = X_1d.tolist()
            result["loss_line_1d"] = loss_line.tolist()
            result["baseline_loss_1d"] = float(baseline_loss_1d) if baseline_loss_1d is not None else float(loss_line.min())
            result["grid_size_1d"] = grid_size
            
            # 如果同时有2D数据，更新mode
            if "loss_grid_2d" in result:
                result["mode"] = "1d+2d"  # 同时包含1D和2D数据
            else:
                result["mode"] = "1d"
                # 如果只有1D数据，则使用X和Y字段（向后兼容）
                result["X"] = X_1d.tolist()
                result["baseline_loss"] = float(baseline_loss_1d) if baseline_loss_1d is not None else float(loss_line.min())
                result["grid_size"] = grid_size

        # 查询并构建3D体积数据（z不为空）
        volume_df = self.conn.execute(
            """
            SELECT x, y, z, loss, epoch
            FROM landscape_points
            WHERE is_trajectory = FALSE AND z IS NOT NULL
            ORDER BY epoch, x, y, z
        """
        ).df()

        if len(volume_df) > 0:
            unique_x = sorted(volume_df["x"].unique())
            unique_y = sorted(volume_df["y"].unique())
            unique_z = sorted(volume_df["z"].unique())

            nx = len(unique_x)
            ny = len(unique_y)
            nz = len(unique_z)

            X3 = np.zeros((nx, ny, nz))
            Y3 = np.zeros((nx, ny, nz))
            Z3 = np.zeros((nx, ny, nz))
            loss_grid_3d = np.zeros((nx, ny, nz))

            for _, row in volume_df.iterrows():
                i = unique_x.index(row["x"])
                j = unique_y.index(row["y"])
                k = unique_z.index(row["z"])
                X3[i, j, k] = row["x"]
                Y3[i, j, k] = row["y"]
                Z3[i, j, k] = row["z"]
                loss_grid_3d[i, j, k] = row["loss"]

            result["Z"] = Z3.tolist()
            result["loss_grid_3d"] = loss_grid_3d.tolist()
            # Also export 1D axes so the frontend can build correctly-scaled slice grids.
            # (Surface X/Y can be a different resolution than volume sampling.)
            result["volume_x"] = [float(v) for v in unique_x]
            result["volume_y"] = [float(v) for v in unique_y]
            result["volume_z"] = [float(v) for v in unique_z]

        # 添加轨迹数据
        if len(trajectory_df) > 0:
            result["trajectory_data"] = {
                "traj_1": trajectory_df["x"].tolist(),
                "traj_2": trajectory_df["y"].tolist(),
                "traj_3": trajectory_df["z"].tolist(),
                "epochs": trajectory_df["epoch"].tolist(),
            }
            # 如果轨迹数据中有loss值，也导出
            if "loss" in trajectory_df.columns:
                losses = trajectory_df["loss"].tolist()
                # 过滤掉None值，保持与epochs的对应关系
                if any(l is not None for l in losses):
                    result["trajectory_data"]["losses"] = losses
            
            # 从metadata中读取val_losses（如果存在）
            try:
                val_losses_result = self.conn.execute("""
                    SELECT value FROM metadata WHERE key = 'trajectory_val_losses'
                """).fetchone()
                if val_losses_result:
                    import json
                    val_losses = json.loads(val_losses_result[0])
                    if any(vl is not None for vl in val_losses):
                        result["trajectory_data"]["val_losses"] = val_losses
            except Exception:
                pass
        
        # 添加metadata信息
        metadata = self.get_metadata()
        if metadata:
            result["metadata"] = metadata
        
        # 在导出前清洗 NaN/Inf，避免前端 JSON.parse 失败
        import math
        def _sanitize(obj):
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj

        sanitized = _sanitize(result)
        
        # 保存为JSON（如果提供路径）
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(sanitized, f, indent=2)
        
        return sanitized

