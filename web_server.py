#!/usr/bin/env python3
"""
视频AI智能识别及预警管理系统 - Web管理服务端
Flask + SQLite + Bootstrap + ECharts 数据大屏
"""

# --- 标准库导入 ---
import os
import sys
import json
import time
import sqlite3
import hashlib
import secrets
import logging
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO

# --- Flask 框架及相关工具导入 ---
from flask import (
    Flask, render_template_string, request, redirect, url_for,
    session, jsonify, send_from_directory, g, flash, make_response
)

# 配置日志：INFO 级别，带时间戳和级别标签
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WebServer")

# 基础路径与数据库路径
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "flame_system.db"
# 上传文件目录，递归创建 pictures / videos / logo 子目录
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR / "pictures").mkdir(exist_ok=True)
(UPLOAD_DIR / "videos").mkdir(exist_ok=True)
(UPLOAD_DIR / "logo").mkdir(exist_ok=True)

# 创建 Flask 应用实例，使用随机生成的密钥用于 session 加密
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


def get_db():
    """获取/创建当前请求的 SQLite 数据库连接，缓存在 Flask g 对象中。
    
    Returns:
        sqlite3.Connection: 数据库连接对象（row_factory 为 sqlite3.Row，启用 WAL 模式和外键约束）
    """
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row  # 允许通过列名访问查询结果
        g.db.execute("PRAGMA journal_mode=WAL")  # 启用 Write-Ahead Logging，提高并发性能
        g.db.execute("PRAGMA foreign_keys=ON")    # 强制外键约束检查
    return g.db


@app.teardown_appcontext
def close_db(exception):
    """应用上下文销毁时自动关闭数据库连接，防止连接泄漏。
    
    Args:
        exception: Flask 传递的异常对象（正常请求时为空）
    """
    db = g.pop("db", None)
    if db:
        db.close()


def init_db():
    """初始化数据库：创建所有业务表结构并执行字段兼容性迁移，最后调用 seed_data 填充初始数据。"""
    db = sqlite3.connect(str(DB_PATH))
    db.executescript("""
-- 系统配置表：存储站点名称、检测阈值、心跳间隔等全局参数
CREATE TABLE IF NOT EXISTS T_Site (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT DEFAULT '火焰预警平台',
    SiteName TEXT DEFAULT '火焰预警平台',
    Logo TEXT,
    thresh REAL DEFAULT 0.35,          -- 火焰检测置信度阈值
    width REAL DEFAULT 640,            -- 视频帧宽度
    height REAL DEFAULT 480,           -- 视频帧高度
    video_times REAL DEFAULT 5,        -- 录像时长（秒）
    heartBeat REAL DEFAULT 1,          -- 心跳间隔（分钟）
    exception_times REAL DEFAULT 5     -- 异常报警触发次数
);

-- 角色表
CREATE TABLE IF NOT EXISTS T_Role (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Description TEXT,
    IsDelete INTEGER DEFAULT 0    -- 软删除标记：0=未删除，1=已删除
);

-- 权限表：角色-权限多对多关联
CREATE TABLE IF NOT EXISTS T_Authority (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    RoleId INTEGER NOT NULL,
    Authority TEXT NOT NULL,
    FOREIGN KEY (RoleId) REFERENCES T_Role(Id)
);

-- 部门/分支机构表：自引用树形结构
CREATE TABLE IF NOT EXISTS T_Branch (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    ParentId INTEGER DEFAULT 0,   -- 上级部门 ID，0 表示顶层
    CreateTime TEXT,
    CreateBy INTEGER,
    Remark TEXT
);

-- 区域表
CREATE TABLE IF NOT EXISTS T_Area (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Remark TEXT
);

-- 用户表
CREATE TABLE IF NOT EXISTS T_User (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Account TEXT UNIQUE,
    Name TEXT NOT NULL,
    AreaId INTEGER,
    BranchId INTEGER,
    Password TEXT NOT NULL,
    CreateTime TEXT,
    CreateBy INTEGER,
    Remark TEXT,
    FOREIGN KEY (BranchId) REFERENCES T_Branch(Id),
    FOREIGN KEY (AreaId) REFERENCES T_Area(Id)
);

-- 用户角色关联表：支持用户绑定多个角色
CREATE TABLE IF NOT EXISTS T_UserRole (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    UserId INTEGER NOT NULL,
    RoleId INTEGER NOT NULL,
    IsDefault TEXT DEFAULT 'isdefault',
    CreateTime TEXT,
    IsDeleted TEXT DEFAULT 'undeleted',   -- 软删除：'undeleted'/'deleted'
    FOREIGN KEY (UserId) REFERENCES T_User(Id),
    FOREIGN KEY (RoleId) REFERENCES T_Role(Id)
);

-- 数据字典表：键值对存储，用于下拉选项等可配置枚举
CREATE TABLE IF NOT EXISTS T_Dictionary (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Key TEXT NOT NULL,
    Value TEXT NOT NULL,
    Remark TEXT
);

-- AI 分析盒（边缘设备）表
CREATE TABLE IF NOT EXISTS T_Device (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    MAC TEXT,                       -- MAC 地址
    Longitude TEXT,
    Latitude TEXT,
    Address TEXT,
    AreaId INTEGER,
    ModelPerson TEXT,
    ModelInfo TEXT,                 -- 模型信息（如 YOLOv11）
    Maintainer TEXT,
    CreateTime TEXT,
    StructuralInfo TEXT,
    DetailInfo TEXT,
    LastConnectTime TEXT,           -- 最后心跳时间
    AutoGenerateError TEXT DEFAULT 'no',  -- 是否自动生成错误记录
    Remark TEXT,
    FOREIGN KEY (AreaId) REFERENCES T_Area(Id)
);

-- 摄像头表
CREATE TABLE IF NOT EXISTS T_Camera (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    IP TEXT,
    MAC TEXT,
    CameraUrl TEXT,                 -- RTSP/RTMP 视频流地址
    Name TEXT,
    Longitude TEXT,
    Latitude TEXT,
    AreaId INTEGER,
    Type TEXT,                      -- 摄像头类型（海康、大华等）
    InstallTime TEXT,
    BandWidth REAL,
    Maintainer TEXT,
    DeviceId INTEGER,               -- 所属 AI 分析盒 ID
    Remark TEXT,
    FOREIGN KEY (AreaId) REFERENCES T_Area(Id),
    FOREIGN KEY (DeviceId) REFERENCES T_Device(Id)
);

-- 火焰检测结果表（核心业务表）
CREATE TABLE IF NOT EXISTS T_DetectResult (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Longitude TEXT,
    Latitude TEXT,
    Location TEXT,
    Picture TEXT,                   -- 报警截图路径
    VideoUrl TEXT,                  -- 报警录像路径
    AreaId INTEGER,
    CreatTime TEXT,
    CameraId INTEGER,
    DeviceId INTEGER,
    Status TEXT DEFAULT '1',        -- 状态：'1'=待处理，'2'=已处理（待审核），'3'=已审核
    OperateUserId INTEGER,          -- 处理人 ID
    OperateTime TEXT,
    UrgencyDegree TEXT,             -- 紧急程度
    OperateResult TEXT,             -- 处理结果
    Description TEXT,
    AuditUserId INTEGER,            -- 审核人 ID
    AuditTime TEXT,
    Remark TEXT,
    FOREIGN KEY (AreaId) REFERENCES T_Area(Id),
    FOREIGN KEY (CameraId) REFERENCES T_Camera(Id),
    FOREIGN KEY (DeviceId) REFERENCES T_Device(Id),
    FOREIGN KEY (OperateUserId) REFERENCES T_User(Id),
    FOREIGN KEY (AuditUserId) REFERENCES T_User(Id)
);

-- 摄像头错误日志表
CREATE TABLE IF NOT EXISTS T_CameraError (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    CameraId INTEGER,
    MAC TEXT,
    CreateTime TEXT,
    ErrorCode TEXT,
    ErrorMsg TEXT,
    Remark TEXT,
    FOREIGN KEY (CameraId) REFERENCES T_Camera(Id)
);

-- 设备错误日志表
CREATE TABLE IF NOT EXISTS T_DeviceError (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    DeviceId INTEGER,
    MAC TEXT,
    CreateTime TEXT,
    ErrorCode TEXT,
    ErrorMsg TEXT,
    Remark TEXT,
    FOREIGN KEY (DeviceId) REFERENCES T_Device(Id)
);

-- 操作日志表：记录用户对系统的增删改操作
CREATE TABLE IF NOT EXISTS T_OperateLog (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    MenuName TEXT,                  -- 操作菜单名称
    Type TEXT,                      -- 操作类型：增加/修改/删除
    ContentNew TEXT,                -- 新内容（JSON 序列化）
    ContentOld TEXT,                -- 旧内容
    CreateTime TEXT,
    UserId INTEGER,
    Remark TEXT,
    FOREIGN KEY (UserId) REFERENCES T_User(Id)
);

-- 用户登录日志表
CREATE TABLE IF NOT EXISTS T_UserLoginLog (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    UserId INTEGER NOT NULL,
    LoginTime TEXT NOT NULL,
    LoginInIp TEXT,                 -- 登录 IP
    LoginType TEXT NOT NULL,        -- 登录方式
    FOREIGN KEY (UserId) REFERENCES T_User(Id)
);
""")
    db.commit()
    # 尝试为旧版本数据库添加新增字段（如果字段已存在则忽略异常）
    try:
        db.execute("ALTER TABLE T_DetectResult ADD COLUMN Confidence REAL DEFAULT 0.0")
        db.commit()
    except sqlite3.OperationalError:
        pass
    try:
        db.execute("ALTER TABLE T_Camera ADD COLUMN WsPort INTEGER DEFAULT 9999")
        db.commit()
    except sqlite3.OperationalError:
        pass
    # Force updating any camera ports still defaulting to 9999 or NULL to be distinct
    try:
        db.execute("UPDATE T_Camera SET WsPort = 9990 + Id WHERE WsPort IS NULL OR WsPort = 9999")
        db.commit()
    except Exception:
        pass

    # Auto-migrate: update existing T_DetectResult.CreatTime to recent/current times if they are stale
    try:
        c = db.execute("SELECT COUNT(*) FROM T_DetectResult").fetchone()
        if c and c[0] > 0:
            latest_row = db.execute("SELECT MAX(CreatTime) FROM T_DetectResult").fetchone()
            if latest_row and latest_row[0]:
                latest_time = datetime.strptime(latest_row[0], "%Y-%m-%d %H:%M:%S")
                time_diff = datetime.now() - latest_time
                if time_diff.days > 2:
                    rows = db.execute("SELECT Id, CreatTime FROM T_DetectResult").fetchall()
                    for r in rows:
                        old_time = datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S")
                        new_time = old_time + time_diff
                        new_time_str = new_time.strftime("%Y-%m-%d %H:%M:%S")
                        db.execute("UPDATE T_DetectResult SET CreatTime = ? WHERE Id = ?", (new_time_str, r[0]))
                    db.commit()
    except Exception as e:
        print(f"Failed to auto-migrate database timestamps: {e}")

    try:
        db.execute("UPDATE T_DetectResult SET OperateResult = NULL WHERE Status = '1'")
        db.commit()
    except Exception as e:
        print(f"Failed to clean up pending operate result database records: {e}")

    db.close()
    seed_data()  # 初始化完成后填充种子数据


def hash_pwd(pwd):
    """使用 SHA256 算法对明文密码进行哈希处理。
    
    Args:
        pwd (str): 明文密码
        
    Returns:
        str: 十六进制编码的 SHA256 哈希值
    """
    return hashlib.sha256(pwd.encode()).hexdigest()


def seed_data():
    """插入系统初始种子数据（仅在数据库为空时执行）。
    
    包括：默认系统配置、三种角色（超级管理员/处理人/审核人）及其权限、
    默认区域、部门、用户账号（admin/chuli001/shenhe001）、数据字典项、
    示例 AI 分析盒和摄像头。
    """
    db = sqlite3.connect(str(DB_PATH))
    c = db.execute("SELECT COUNT(*) FROM T_User").fetchone()
    if c and c[0] > 0:
        db.close()
        return  # 已有用户数据，跳过种子数据初始化

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 插入默认系统配置
    db.execute("INSERT INTO T_Site (Name, SiteName, thresh, width, height, video_times, heartBeat, exception_times) VALUES (?,?,?,?,?,?,?,?)",
               ("火焰检测预警平台", "火焰预警平台", 0.35, 640, 480, 5, 1, 5))

    # 创建三种预设角色
    db.execute("INSERT INTO T_Role (Id, Name, Description) VALUES (1,'超级管理员','系统最高权限')")
    db.execute("INSERT INTO T_Role (Id, Name, Description) VALUES (2,'处理人','事件处理人员')")
    db.execute("INSERT INTO T_Role (Id, Name, Description) VALUES (3,'审核人','事件审核人员')")

    # 为超级管理员分配所有功能权限
    for auth in ["system_config","department","user","role","device","camera","alarm","audit","log","dashboard","dictionary"]:
        db.execute("INSERT INTO T_Authority (RoleId, Authority) VALUES (1,?)", (auth,))
    # 处理人权限：报警、摄像头、设备、仪表盘
    for auth in ["alarm","camera","device","dashboard"]:
        db.execute("INSERT INTO T_Authority (RoleId, Authority) VALUES (2,?)", (auth,))
    # 审核人权限：报警、审核、仪表盘
    for auth in ["alarm","audit","dashboard"]:
        db.execute("INSERT INTO T_Authority (RoleId, Authority) VALUES (3,?)", (auth,))

    # 插入默认区域和部门
    db.execute("INSERT INTO T_Area (Id, Name) VALUES (1,'重庆市'),(2,'北京市'),(3,'上海市'),(4,'广州市'),(5,'成都市')")
    db.execute("INSERT INTO T_Branch (Id, Name, ParentId, CreateTime) VALUES (1,'总公司',0,?),(2,'重庆分公司',1,?),(3,'技术部',1,?),(4,'运维部',1,?)", (now, now, now, now))

    # 创建默认用户（密码统一为 123456 的 SHA256 哈希）
    db.execute("INSERT INTO T_User (Id, Account, Name, AreaId, BranchId, Password, CreateTime) VALUES (1,'admin','系统管理员',1,1,?,?)",
               (hash_pwd("123456"), now))
    db.execute("INSERT INTO T_User (Id, Account, Name, AreaId, BranchId, Password, CreateTime) VALUES (2,'chuli001','张处理',1,3,?,?)",
               (hash_pwd("123456"), now))
    db.execute("INSERT INTO T_User (Id, Account, Name, AreaId, BranchId, Password, CreateTime) VALUES (3,'shenhe001','李审核',1,4,?,?)",
               (hash_pwd("123456"), now))

    # 绑定用户与角色
    db.execute("INSERT INTO T_UserRole (UserId, RoleId, IsDefault, CreateTime, IsDeleted) VALUES (1,1,'isdefault',?,'undeleted')", (now,))
    db.execute("INSERT INTO T_UserRole (UserId, RoleId, IsDefault, CreateTime, IsDeleted) VALUES (2,2,'isdefault',?,'undeleted')", (now,))
    db.execute("INSERT INTO T_UserRole (UserId, RoleId, IsDefault, CreateTime, IsDeleted) VALUES (3,3,'isdefault',?,'undeleted')", (now,))

    # 填充数据字典（区域、紧急程度、处理结果、摄像头类型、错误代码等枚举值）
    dict_data = [
        ("AreaType", "重庆市"), ("AreaType", "北京市"), ("AreaType", "上海市"), ("AreaType", "广州市"), ("AreaType", "成都市"),
        ("UrgencyDegree", "一般"), ("UrgencyDegree", "紧急"), ("UrgencyDegree", "非常紧急"),
        ("OperateResult", "已处理"), ("OperateResult", "误报"), ("OperateResult", "待观察"), ("OperateResult", "无需处理"),
        ("CameraType", "海康威视"), ("CameraType", "大华"), ("CameraType", "宇视"), ("CameraType", "其他"),
        ("ErrorCode", "网络故障"), ("ErrorCode", "图像质量差"), ("ErrorCode", "设备离线"),
    ]
    for key, val in dict_data:
        db.execute("INSERT INTO T_Dictionary (Key, Value) VALUES (?,?)", (key, val))

    # 插入示例 AI 分析盒设备
    db.execute("INSERT INTO T_Device (Id, MAC, Longitude, Latitude, Address, AreaId, ModelInfo, CreateTime, LastConnectTime) VALUES (1,'AAABBBCCCDDD','106.529813','29.452537','重庆理工大学花溪校区',1,'YOLOv11-Fire',?,?)", (now, now))
    db.execute("INSERT INTO T_Device (Id, MAC, Longitude, Latitude, Address, AreaId, ModelInfo, CreateTime, LastConnectTime) VALUES (2,'EEEFFFGGGHHH','106.542236','29.606703','重庆理工大学杨家坪校区',1,'YOLOv11-Fire',?,?)", (now, now))

    # 填充摄像头故障和AI分析盒故障初始数据，满足系统故障展示要求
    db.execute("INSERT INTO T_DeviceError (DeviceId, MAC, CreateTime, ErrorCode, ErrorMsg) VALUES (1, 'AAABBBCCCDDD', ?, '算力异常', 'NPU核心温度 85℃，推理帧率降为 3 FPS，低于系统设定的 15 FPS 阈值')", (now,))
    db.execute("INSERT INTO T_DeviceError (DeviceId, MAC, CreateTime, ErrorCode, ErrorMsg) VALUES (1, 'AAABBBCCCDDD', ?, '算法崩溃', 'YOLOv11 推理线程异常退出 (exit status 139: segmentation fault)')", (now,))
    db.execute("INSERT INTO T_CameraError (CameraId, MAC, CreateTime, ErrorCode, ErrorMsg) VALUES (1, '127.0.0.1:9991', ?, '图像质量差', '视频源光照过低或被遮挡，图像置信度降至 25%')", (now,))
    db.execute("INSERT INTO T_CameraError (CameraId, MAC, CreateTime, ErrorCode, ErrorMsg) VALUES (2, '127.0.0.1:9992', ?, '网络故障', '网络延迟异常 (Ping > 500ms)，触发摄像头丢包故障')", (now,))

    # 系统初始默认不预置任何摄像头，启动后将通过前端局域网 WebSocket 端口自动扫描和发现注册

    db.commit()
    db.close()
    logger.info("Database seeded with initial data")


def check_and_log_faults(db):
    """自适应在线/离线故障检测：检查 T_Device 表中设备的最后连接时间，
    如果心跳超时超过 15 秒且设备为 'no' (未产生离线错误)，则自动生成离线记录。
    """
    try:
        now = datetime.now()
        threshold = (now - timedelta(seconds=15)).strftime("%Y-%m-%d %H:%M:%S")
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # 查找超时的在线设备
        offline_devices = db.execute(
            "SELECT * FROM T_Device WHERE LastConnectTime < ? AND AutoGenerateError = 'no'",
            (threshold,)
        ).fetchall()
        
        for dev in offline_devices:
            dev_id = dev["Id"]
            mac = dev["MAC"] or "unknown"
            addr = dev["Address"] or "未知分析盒"
            
            # 记录分析盒故障日志
            db.execute(
                "INSERT INTO T_DeviceError (DeviceId, MAC, CreateTime, ErrorCode, ErrorMsg) VALUES (?,?,?,?,?)",
                (dev_id, mac, now_str, "设备离线", f"AI分析盒 [{addr}] 与云端失去心跳连接 (超时超过15秒)")
            )
            
            # 查找该分析盒下的摄像头并记录故障
            cameras = db.execute("SELECT * FROM T_Camera WHERE DeviceId = ?", (dev_id,)).fetchall()
            for cam in cameras:
                db.execute(
                    "INSERT INTO T_CameraError (CameraId, MAC, CreateTime, ErrorCode, ErrorMsg) VALUES (?,?,?,?,?)",
                    (cam["Id"], f"Port:{cam['WsPort']}", now_str, "设备离线", f"监控流中断 (AI分析盒离线)")
                )
                
            # 标记为已处理错误，避免重复产生
            db.execute("UPDATE T_Device SET AutoGenerateError = 'yes' WHERE Id = ?", (dev_id,))
        db.commit()
    except Exception as e:
        logger.error(f"check_and_log_faults error: {e}")


# --- 认证与授权装饰器及辅助函数 ---

def login_required(f):
    """登录验证装饰器：未登录用户自动跳转到登录页面。
    
    Args:
        f: 被装饰的视图函数
        
    Returns:
        function: 包装后的函数，仅在 session 中存在 user_id 时才执行原函数
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """管理员权限装饰器：仅允许 role_id==1（超级管理员）访问，否则提示权限不足并跳转仪表盘。
    
    Args:
        f: 被装饰的视图函数
        
    Returns:
        function: 包装后的函数，仅允许超级管理员执行
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role_id") != 1:
            flash("权限不足")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


def get_current_user():
    """获取当前登录用户信息，联表查询获取角色名称。
    
    Returns:
        sqlite3.Row 或 None: 当前登录用户记录（含 RoleName 字段），未登录则返回 None
    """
    if "user_id" not in session:
        return None
    db = get_db()
    return db.execute("SELECT u.*, r.Name as RoleName FROM T_User u LEFT JOIN T_UserRole ur ON u.Id=ur.UserId LEFT JOIN T_Role r ON ur.RoleId=r.Id WHERE u.Id=?", (session["user_id"],)).fetchone()


def add_log(menu_name, op_type, content_new, content_old=""):
    """记录用户操作日志到 T_OperateLog 表。
    
    Args:
        menu_name (str): 操作菜单名称（如"用户管理"、"系统配置"等）
        op_type (str): 操作类型（如"增加"、"修改"、"删除"）
        content_new: 新内容（将被转换为字符串并截断至500字符）
        content_old: 旧内容（可选，同样截断至500字符）
    """
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("INSERT INTO T_OperateLog (MenuName, Type, ContentNew, ContentOld, CreateTime, UserId) VALUES (?,?,?,?,?,?)",
               (menu_name, op_type, str(content_new)[:500], str(content_old)[:500], now, session.get("user_id", 0)))
    db.commit()


# --- 路由：登录 / 登出 ---

@app.route("/")
def index():
    """根路径重定向到仪表盘页面。"""
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    """登录页面：GET 展示登录表单，POST 验证账号密码并写入 session。
    
    登录成功记录登录日志到 T_UserLoginLog，失败则 flash 提示错误。
    """
    if request.method == "POST":
        account = request.form.get("account", "")
        password = request.form.get("password", "")
        db = get_db()
        # 联表查询用户及角色信息（仅查询未删除的角色绑定）
        user = db.execute("SELECT u.*, ur.RoleId FROM T_User u LEFT JOIN T_UserRole ur ON u.Id=ur.UserId WHERE u.Account=? AND ur.IsDeleted='undeleted'",
                          (account,)).fetchone()
        if user and user["Password"] == hash_pwd(password):
            # 密码验证通过，设置 session
            session["user_id"] = user["Id"]
            session["user_name"] = user["Name"]
            session["role_id"] = user["RoleId"]
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db.execute("INSERT INTO T_UserLoginLog (UserId, LoginTime, LoginInIp, LoginType) VALUES (?,?,?,?)",
                       (user["Id"], now, request.remote_addr, "后台登陆"))
            db.commit()
            flash(f"欢迎回来, {user['Name']}!")
            return redirect(url_for("dashboard"))
        flash("账号或密码错误")
    return render_template_string(LOGIN_TEMPLATE)


@app.route("/logout")
def logout():
    """登出：清除当前 session 并重定向到登录页。"""
    session.clear()
    return redirect(url_for("login_page"))


# --- 路由：数据大屏仪表盘 ---

@app.route("/dashboard")
@login_required
def dashboard():
    """数据大屏仪表盘：展示今日/本周/本月/本年报警统计、报警状态分布、
    最新报警列表、各区域月排名、设备与摄像头在地图上的分布等。
    
    Returns:
        str: 渲染后的 HTML 页面
    """
    user = get_current_user()
    db = get_db()
    check_and_log_faults(db)

    # 全局统计：总报警数、待处理数、设备总数
    total_alarms = db.execute("SELECT COUNT(*) as c FROM T_DetectResult").fetchone()["c"]
    pending_alarms = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE Status='1'").fetchone()["c"]
    total_devices = db.execute("SELECT COUNT(*) as c FROM T_Device").fetchone()["c"]

    # 计算不同时间段的报警数量
    today_start = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S")

    today_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE CreatTime >= ?", (today_start,)).fetchone()["c"]
    week_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE CreatTime > ?", (week_ago,)).fetchone()["c"]
    month_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE CreatTime > ?", (month_ago,)).fetchone()["c"]
    year_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE CreatTime > ?", (year_ago,)).fetchone()["c"]

    # 按处理结果分类统计：确认真火警 / 误报 / 漏报
    true_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE OperateResult='火灾已确认并报警'").fetchone()["c"]
    false_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE OperateResult='误报无需处理'").fetchone()["c"]
    missed_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE OperateResult='漏报记录'").fetchone()["c"]

    # 最新报警列表（未处理优先，置信度降序，时间降序）
    recent_alarms = [dict(r) for r in db.execute(
        "SELECT dr.*, c.Name as CameraName, a.Name as AreaName, u.Name as OperatorName FROM T_DetectResult dr LEFT JOIN T_Camera c ON dr.CameraId=c.Id LEFT JOIN T_Area a ON dr.AreaId=a.Id LEFT JOIN T_User u ON dr.OperateUserId=u.Id ORDER BY (CASE WHEN dr.Status='1' THEN 1 WHEN dr.OperateResult='误报无需处理' THEN 4 WHEN dr.OperateResult='漏报记录' THEN 3 ELSE 2 END) ASC, dr.Confidence DESC, dr.CreatTime DESC LIMIT 30").fetchall()]

    # 最早报警时间（用于数据大屏时间轴起始点）
    earliest_row = db.execute("SELECT MIN(CreatTime) as m FROM T_DetectResult").fetchone()
    earliest_time = earliest_row["m"] if earliest_row and earliest_row["m"] else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 各区域月度报警排名（Top 5）
    monthly_ranking = [dict(r) for r in db.execute(
        "SELECT COALESCE(NULLIF(dr.Location, ''), '未知区域') as name, COUNT(*) as count FROM T_DetectResult dr WHERE dr.CreatTime > ? GROUP BY dr.Location ORDER BY count DESC LIMIT 5", (month_ago,)).fetchall()]
    max_rank = max([r["count"] for r in monthly_ranking]) if monthly_ranking else 1  # 排名中的最大值，用于进度条比例

    # 所有摄像头列表（用于地图标注）
    cameras = [dict(r) for r in db.execute("SELECT * FROM T_Camera ORDER BY Id").fetchall()]

    # 在线/离线设备统计 (心跳周期 30s，允许 60s 宽限期)
    online_threshold = (datetime.now() - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")
    online_count = db.execute("SELECT COUNT(*) as c FROM T_Device WHERE LastConnectTime >= ?", (online_threshold,)).fetchone()["c"]
    offline_count = max(0, total_devices - online_count)

    # AI处理率
    ai_rate = round(((total_alarms - pending_alarms) / total_alarms * 100) if total_alarms > 0 else 100, 1)

    return render_template_string(DASHBOARD_TEMPLATE, user=user,
                                  total_alarms=total_alarms, pending_alarms=pending_alarms,
                                  total_devices=total_devices,
                                  today_count=today_count, week_count=week_count,
                                  month_count=month_count, year_count=year_count,
                                  true_count=true_count, false_count=false_count,
                                  missed_count=missed_count,
                                  online_count=online_count, offline_count=offline_count,
                                  ai_rate=ai_rate,
                                  recent_alarms=recent_alarms,
                                  monthly_ranking=monthly_ranking, max_rank=max_rank,
                                  earliest_time=earliest_time,
                                  cameras=cameras)


# --- 路由：系统配置 ---

@app.route("/admin/config", methods=["GET", "POST"])
@login_required
@admin_required
def system_config():
    """系统配置页面：GET 展示当前配置，POST 更新系统参数（站点名、阈值、尺寸等）。
    
    仅超级管理员可访问。
    """
    db = get_db()
    if request.method == "POST":
        data = {k: request.form[k] for k in ["Name", "SiteName", "thresh", "width", "height", "video_times", "heartBeat", "exception_times"]}
        db.execute("UPDATE T_Site SET Name=?, SiteName=?, thresh=?, width=?, height=?, video_times=?, heartBeat=?, exception_times=? WHERE Id=1",
                   (data["Name"], data["SiteName"], float(data["thresh"]), float(data["width"]), float(data["height"]), float(data["video_times"]), float(data["heartBeat"]), float(data["exception_times"])))
        db.commit()
        add_log("系统配置", "修改", data)  # 记录操作日志
        flash("系统配置已更新")
    site = db.execute("SELECT * FROM T_Site WHERE Id=1").fetchone()
    return render_template_string(CONFIG_TEMPLATE, user=get_current_user(), site=dict(site) if site else {})


# --- 路由：部门管理 ---

@app.route("/admin/branch")
@login_required
@admin_required
def branch_list():
    """部门管理列表页面：展示所有部门信息。"""
    db = get_db()
    branches = [dict(r) for r in db.execute("SELECT * FROM T_Branch ORDER BY Id").fetchall()]
    return render_template_string(BRANCH_TEMPLATE, user=get_current_user(), branches=branches)


@app.route("/admin/branch/add", methods=["POST"])
@login_required
@admin_required
def branch_add():
    """新增部门：受理 POST 表单，向 T_Branch 插入新记录。"""
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("INSERT INTO T_Branch (Name, ParentId, CreateTime, CreateBy, Remark) VALUES (?,?,?,?,?)",
               (request.form["Name"], int(request.form.get("ParentId", 0)), now, session["user_id"], request.form.get("Remark", "")))
    db.commit()
    add_log("部门管理", "增加", dict(request.form))
    flash("部门已添加")
    return redirect(url_for("branch_list"))


@app.route("/admin/branch/edit/<int:bid>", methods=["POST"])
@login_required
@admin_required
def branch_edit(bid):
    """编辑部门：根据部门 ID 更新名称、上级部门、备注等信息。
    
    Args:
        bid: 部门 ID
    """
    db = get_db()
    db.execute("UPDATE T_Branch SET Name=?, ParentId=?, Remark=? WHERE Id=?",
               (request.form["Name"], int(request.form.get("ParentId", 0)), request.form.get("Remark", ""), bid))
    db.commit()
    add_log("部门管理", "修改", dict(request.form))
    flash("部门已更新")
    return redirect(url_for("branch_list"))


@app.route("/admin/branch/delete/<int:bid>")
@login_required
@admin_required
def branch_delete(bid):
    """删除部门：物理删除指定 ID 的部门记录。
    
    Args:
        bid: 部门 ID
    """
    db = get_db()
    db.execute("DELETE FROM T_Branch WHERE Id=?", (bid,))
    db.commit()
    add_log("部门管理", "删除", {"Id": bid})
    flash("部门已删除")
    return redirect(url_for("branch_list"))


# --- 路由：用户管理 ---

@app.route("/admin/user")
@login_required
@admin_required
def user_list():
    """用户管理列表页面：联表查询用户信息，含部门、区域、角色名称。"""
    db = get_db()
    users = [dict(r) for r in db.execute(
        "SELECT u.*, b.Name as BranchName, a.Name as AreaName, r.Name as RoleName FROM T_User u LEFT JOIN T_Branch b ON u.BranchId=b.Id LEFT JOIN T_Area a ON u.AreaId=a.Id LEFT JOIN T_UserRole ur ON u.Id=ur.UserId LEFT JOIN T_Role r ON ur.RoleId=r.Id WHERE ur.IsDeleted='undeleted' ORDER BY u.Id").fetchall()]
    branches = [dict(r) for r in db.execute("SELECT * FROM T_Branch").fetchall()]
    areas = [dict(r) for r in db.execute("SELECT Id,Value as Name FROM T_Dictionary WHERE Key='AreaType'").fetchall()]
    roles = [dict(r) for r in db.execute("SELECT * FROM T_Role WHERE IsDelete=0").fetchall()]
    return render_template_string(USER_TEMPLATE, user=get_current_user(), users=users, branches=branches, areas=areas, roles=roles)


@app.route("/admin/user/add", methods=["POST"])
@login_required
@admin_required
def user_add():
    """新增用户：创建用户记录并绑定角色关联。
    
    密码经过 SHA256 哈希后存储，操作日志中排除密码字段。
    """
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pwd = hash_pwd(request.form["Password"])
    # 插入用户基本信息
    db.execute("INSERT INTO T_User (Account, Name, AreaId, BranchId, Password, CreateTime, CreateBy) VALUES (?,?,?,?,?,?,?)",
               (request.form["Account"], request.form["Name"], int(request.form.get("AreaId", 1)), int(request.form.get("BranchId", 1)), pwd, now, session["user_id"]))
    uid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    # 创建用户角色绑定
    db.execute("INSERT INTO T_UserRole (UserId, RoleId, IsDefault, CreateTime, IsDeleted) VALUES (?,?,?,?,?)",
               (uid, int(request.form.get("RoleId", 2)), request.form.get("IsDefault", "isdefault"), now, "undeleted"))
    db.commit()
    add_log("用户管理", "增加", {k: v for k, v in request.form.items() if k != "Password"})
    flash("用户已添加")
    return redirect(url_for("user_list"))


@app.route("/admin/user/edit/<int:uid>", methods=["POST"])
@login_required
@admin_required
def user_edit(uid):
    """编辑用户：更新用户基本信息和角色绑定。
    
    如果未提供新密码则保留原密码不变（仅更新其他字段）。
    
    Args:
        uid: 用户 ID
    """
    db = get_db()
    pwd = request.form.get("Password", "")
    if pwd:
        # 提供了新密码，重新哈希后更新
        db.execute("UPDATE T_User SET Account=?, Name=?, AreaId=?, BranchId=?, Password=?, Remark=? WHERE Id=?",
                   (request.form["Account"], request.form["Name"], int(request.form.get("AreaId", 1)), int(request.form.get("BranchId", 1)), hash_pwd(pwd), request.form.get("Remark", ""), uid))
    else:
        # 未提供密码，忽略密码字段
        db.execute("UPDATE T_User SET Account=?, Name=?, AreaId=?, BranchId=?, Remark=? WHERE Id=?",
                   (request.form["Account"], request.form["Name"], int(request.form.get("AreaId", 1)), int(request.form.get("BranchId", 1)), request.form.get("Remark", ""), uid))
    if request.form.get("RoleId"):
        # 更新角色绑定
        db.execute("UPDATE T_UserRole SET RoleId=? WHERE UserId=? AND IsDeleted='undeleted'",
                   (int(request.form["RoleId"]), uid))
    db.commit()
    add_log("用户管理", "修改", {k: v for k, v in request.form.items() if k != "Password"})
    flash("用户已更新")
    return redirect(url_for("user_list"))


@app.route("/admin/user/delete/<int:uid>")
@login_required
@admin_required
def user_delete(uid):
    """删除用户：软删除（标记用户角色绑定为 deleted），不物理删除用户记录。
    
    Args:
        uid: 用户 ID
    """
    db = get_db()
    db.execute("UPDATE T_UserRole SET IsDeleted='deleted' WHERE UserId=?", (uid,))
    db.commit()
    add_log("用户管理", "删除", {"Id": uid})
    flash("用户已删除")
    return redirect(url_for("user_list"))


# --- 路由：角色管理 ---

@app.route("/admin/role")
@login_required
@admin_required
def role_list():
    """角色管理列表页面：展示所有未删除的角色。"""
    db = get_db()
    roles = [dict(r) for r in db.execute("SELECT * FROM T_Role WHERE IsDelete=0").fetchall()]
    return render_template_string(ROLE_TEMPLATE, user=get_current_user(), roles=roles)


@app.route("/admin/role/add", methods=["POST"])
@login_required
@admin_required
def role_add():
    """新增角色：创建角色并批量插入权限记录。"""
    db = get_db()
    db.execute("INSERT INTO T_Role (Name, Description) VALUES (?,?)",
               (request.form["Name"], request.form.get("Description", "")))
    rid = db.execute("SELECT last_insert_rowid()").fetchone()[0]  # 获取新角色 ID
    authorities = request.form.getlist("authorities")  # 多选权限列表
    for auth in authorities:
        db.execute("INSERT INTO T_Authority (RoleId, Authority) VALUES (?,?)", (rid, auth))
    db.commit()
    add_log("角色管理", "增加", dict(request.form))
    flash("角色已添加")
    return redirect(url_for("role_list"))


@app.route("/admin/role/edit/<int:rid>", methods=["POST"])
@login_required
@admin_required
def role_edit(rid):
    """编辑角色：更新角色信息并重建权限（先删后插）。
    
    Args:
        rid: 角色 ID
    """
    db = get_db()
    db.execute("UPDATE T_Role SET Name=?, Description=? WHERE Id=?",
               (request.form["Name"], request.form.get("Description", ""), rid))
    # 先删除该角色的所有旧权限，再插入新权限
    db.execute("DELETE FROM T_Authority WHERE RoleId=?", (rid,))
    authorities = request.form.getlist("authorities")
    for auth in authorities:
        db.execute("INSERT INTO T_Authority (RoleId, Authority) VALUES (?,?)", (rid, auth))
    db.commit()
    add_log("角色管理", "修改", dict(request.form))
    flash("角色已更新")
    return redirect(url_for("role_list"))


@app.route("/admin/role/delete/<int:rid>")
@login_required
@admin_required
def role_delete(rid):
    """删除角色：软删除（设置 IsDelete=1）。
    
    Args:
        rid: 角色 ID
    """
    db = get_db()
    db.execute("UPDATE T_Role SET IsDelete=1 WHERE Id=?", (rid,))
    db.commit()
    add_log("角色管理", "删除", {"Id": rid})
    flash("角色已删除")
    return redirect(url_for("role_list"))


# --- 路由：数据字典管理 ---

@app.route("/admin/dictionary")
@login_required
@admin_required
def dictionary_list():
    """数据字典管理页面：按 Key 分组展示所有字典项。"""
    db = get_db()
    keys = [dict(r) for r in db.execute("SELECT DISTINCT Key FROM T_Dictionary").fetchall()]
    items = [dict(r) for r in db.execute("SELECT * FROM T_Dictionary ORDER BY Key, Id").fetchall()]
    return render_template_string(DICT_TEMPLATE, user=get_current_user(), keys=keys, items=items)


@app.route("/admin/dictionary/add", methods=["POST"])
@login_required
@admin_required
def dictionary_add():
    """新增字典项：向指定 Key 下添加一条 Value 记录。"""
    db = get_db()
    db.execute("INSERT INTO T_Dictionary (Key, Value, Remark) VALUES (?,?,?)",
               (request.form["Key"], request.form["Value"], request.form.get("Remark", "")))
    db.commit()
    flash("字典项已添加")
    return redirect(url_for("dictionary_list"))


@app.route("/admin/dictionary/delete/<int:did>")
@login_required
@admin_required
def dictionary_delete(did):
    """删除字典项：物理删除指定 ID 的字典记录。
    
    Args:
        did: 字典项 ID
    """
    db = get_db()
    db.execute("DELETE FROM T_Dictionary WHERE Id=?", (did,))
    db.commit()
    flash("字典项已删除")
    return redirect(url_for("dictionary_list"))


# --- 路由：AI 分析盒（边缘设备）管理 ---

@app.route("/admin/device")
@login_required
def device_list():
    """AI 分析盒列表页面：展示所有边缘设备及其关联的区域名称。
    
    注意：此页面未加 @admin_required，允许所有登录用户查看。
    """
    db = get_db()
    devices = [dict(r) for r in db.execute(
        "SELECT d.*, a.Name as AreaName FROM T_Device d LEFT JOIN T_Area a ON d.AreaId=a.Id ORDER BY d.Id").fetchall()]
    
    now = datetime.now()
    for d in devices:
        if d.get("LastConnectTime"):
            try:
                lct = datetime.strptime(d["LastConnectTime"], "%Y-%m-%d %H:%M:%S")
                d["status"] = "online" if (now - lct).total_seconds() <= 15 else "offline"
            except Exception:
                d["status"] = "offline"
        else:
            d["status"] = "offline"
            
    areas = [dict(r) for r in db.execute("SELECT Id,Value as Name FROM T_Dictionary WHERE Key='AreaType'").fetchall()]
    return render_template_string(DEVICE_TEMPLATE, user=get_current_user(), devices=devices, areas=areas)


@app.route("/admin/device/add", methods=["POST"])
@login_required
@admin_required
def device_add():
    """新增 AI 分析盒：受理 POST 表单，向 T_Device 插入新记录。"""
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("""INSERT INTO T_Device (MAC, Longitude, Latitude, Address, AreaId, ModelPerson, ModelInfo, Maintainer, CreateTime, StructuralInfo, DetailInfo, Remark)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
               (request.form.get("MAC", ""), request.form.get("Longitude", ""), request.form.get("Latitude", ""),
                request.form.get("Address", ""), int(request.form.get("AreaId", 1)), request.form.get("ModelPerson", ""),
                request.form.get("ModelInfo", ""), request.form.get("Maintainer", ""), now,
                request.form.get("StructuralInfo", ""), request.form.get("DetailInfo", ""), request.form.get("Remark", "")))
    db.commit()
    add_log("AI分析盒管理", "增加", dict(request.form))
    flash("AI分析盒已添加")
    return redirect(url_for("device_list"))


@app.route("/admin/device/edit/<int:did>", methods=["POST"])
@login_required
@admin_required
def device_edit(did):
    """编辑 AI 分析盒：根据设备 ID 更新设备信息。
    
    Args:
        did: 设备 ID
    """
    db = get_db()
    db.execute("""UPDATE T_Device SET MAC=?, Longitude=?, Latitude=?, Address=?, AreaId=?, ModelPerson=?, ModelInfo=?, Maintainer=?, StructuralInfo=?, DetailInfo=?, Remark=? WHERE Id=?""",
               (request.form.get("MAC", ""), request.form.get("Longitude", ""), request.form.get("Latitude", ""),
                request.form.get("Address", ""), int(request.form.get("AreaId", 1)), request.form.get("ModelPerson", ""),
                request.form.get("ModelInfo", ""), request.form.get("Maintainer", ""),
                request.form.get("StructuralInfo", ""), request.form.get("DetailInfo", ""), request.form.get("Remark", ""), did))
    db.commit()
    add_log("AI分析盒管理", "修改", dict(request.form))
    flash("AI分析盒已更新")
    return redirect(url_for("device_list"))


@app.route("/admin/device/delete/<int:did>")
@login_required
@admin_required
def device_delete(did):
    """删除 AI 分析盒：物理删除指定 ID 的设备记录。
    
    Args:
        did: 设备 ID
    """
    db = get_db()
    db.execute("DELETE FROM T_Device WHERE Id=?", (did,))
    db.commit()
    add_log("AI分析盒管理", "删除", {"Id": did})
    flash("AI分析盒已删除")
    return redirect(url_for("device_list"))


# --- 路由：摄像头管理 ---

@app.route("/admin/camera")
@login_required
def camera_list():
    """摄像头列表页面：联表查询摄像头及关联的设备、区域信息。"""
    db = get_db()
    cameras = [dict(r) for r in db.execute(
        "SELECT c.*, a.Name as AreaName, d.MAC as DeviceMAC, d.Address as DeviceAddress, d.LastConnectTime FROM T_Camera c LEFT JOIN T_Area a ON c.AreaId=a.Id LEFT JOIN T_Device d ON c.DeviceId=d.Id ORDER BY c.Id").fetchall()]
    
    now = datetime.now()
    for c in cameras:
        if c.get("LastConnectTime"):
            try:
                lct = datetime.strptime(c["LastConnectTime"], "%Y-%m-%d %H:%M:%S")
                c["status"] = "online" if (now - lct).total_seconds() <= 15 else "offline"
            except Exception:
                c["status"] = "offline"
        else:
            c["status"] = "offline"

    areas = [dict(r) for r in db.execute("SELECT Id,Value as Name FROM T_Dictionary WHERE Key='AreaType'").fetchall()]
    devices = [dict(r) for r in db.execute("SELECT * FROM T_Device").fetchall()]
    return render_template_string(CAMERA_TEMPLATE, user=get_current_user(), cameras=cameras, areas=areas, devices=devices)


@app.route("/admin/camera/add", methods=["POST"])
@login_required
@admin_required
def camera_add():
    """新增摄像头：受理 POST 表单，向 T_Camera 插入新记录。"""
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("""INSERT INTO T_Camera (IP, MAC, CameraUrl, Name, Longitude, Latitude, AreaId, Type, InstallTime, BandWidth, Maintainer, DeviceId, Remark)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
               (request.form.get("IP", ""), request.form.get("MAC", ""), request.form.get("CameraUrl", ""),
                request.form.get("Name", ""), request.form.get("Longitude", ""), request.form.get("Latitude", ""),
                int(request.form.get("AreaId", 1)), request.form.get("Type", ""), now,
                float(request.form.get("BandWidth", 0)), request.form.get("Maintainer", ""),
                int(request.form.get("DeviceId", 1)), request.form.get("Remark", "")))
    db.commit()
    add_log("摄像头管理", "增加", dict(request.form))
    flash("摄像头已添加")
    return redirect(url_for("camera_list"))


@app.route("/admin/camera/edit/<int:cid>", methods=["POST"])
@login_required
@admin_required
def camera_edit(cid):
    """编辑摄像头：根据摄像头 ID 更新摄像头信息。
    
    Args:
        cid: 摄像头 ID
    """
    db = get_db()
    db.execute("""UPDATE T_Camera SET IP=?, MAC=?, CameraUrl=?, Name=?, Longitude=?, Latitude=?, AreaId=?, Type=?, BandWidth=?, Maintainer=?, DeviceId=?, Remark=? WHERE Id=?""",
               (request.form.get("IP", ""), request.form.get("MAC", ""), request.form.get("CameraUrl", ""),
                request.form.get("Name", ""), request.form.get("Longitude", ""), request.form.get("Latitude", ""),
                int(request.form.get("AreaId", 1)), request.form.get("Type", ""),
                float(request.form.get("BandWidth", 0)), request.form.get("Maintainer", ""),
                int(request.form.get("DeviceId", 1)), request.form.get("Remark", ""), cid))
    db.commit()
    add_log("摄像头管理", "修改", dict(request.form))
    flash("摄像头已更新")
    return redirect(url_for("camera_list"))


@app.route("/admin/camera/delete/<int:cid>")
@login_required
@admin_required
def camera_delete(cid):
    """删除摄像头：物理删除指定 ID 的摄像头记录。
    
    Args:
        cid: 摄像头 ID
    """
    db = get_db()
    db.execute("DELETE FROM T_Camera WHERE Id=?", (cid,))
    db.commit()
    add_log("摄像头管理", "删除", {"Id": cid})
    flash("摄像头已删除")
    return redirect(url_for("camera_list"))


# --- 路由：报警事件管理 ---

@app.route("/admin/alarm")
@login_required
def alarm_list():
    """报警事件列表页面：按创建时间降序展示所有检测结果。"""
    db = get_db()
    alarms = [dict(r) for r in db.execute(
        "SELECT dr.*, c.Name as CameraName, a.Name as AreaName, d.Address as DeviceAddress, u.Name as OperatorName FROM T_DetectResult dr LEFT JOIN T_Camera c ON dr.CameraId=c.Id LEFT JOIN T_Area a ON dr.AreaId=a.Id LEFT JOIN T_Device d ON dr.DeviceId=d.Id LEFT JOIN T_User u ON dr.OperateUserId=u.Id ORDER BY dr.CreatTime DESC").fetchall()]
    return render_template_string(ALARM_TEMPLATE, user=get_current_user(), alarms=alarms)


@app.route("/admin/alarm/process/<int:aid>", methods=["POST"])
@login_required
def alarm_process(aid):
    """处理报警事件：更新报警状态为已处理，记录处理人、紧急程度、处理结果等。
    
    Args:
        aid: 报警记录 ID
    """
    user = get_current_user()
    if user and user["RoleName"] == "审核人":
        flash("权限不足：审核人无权处理报警事件")
        ref = request.referrer
        if ref:
            if "/dashboard" in ref or "127.0.0.1" in ref:
                return redirect(url_for("dashboard"))
            elif "/admin/audit" in ref:
                return redirect(url_for("audit_list"))
        return redirect(url_for("alarm_list"))

    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("UPDATE T_DetectResult SET Status='2', OperateUserId=?, OperateTime=?, UrgencyDegree=?, OperateResult=?, Description=? WHERE Id=?",
               (session["user_id"], now, request.form.get("UrgencyDegree", ""), request.form.get("OperateResult", ""), request.form.get("Description", ""), aid))
    db.commit()
    add_log("报警事件", "处理", {"alarm_id": aid, "result": request.form.get("OperateResult", "")})
    flash("事件已处理")
    # 根据来源页面智能跳转回原页面
    ref = request.referrer
    if ref:
        if "/dashboard" in ref:
            return redirect(url_for("dashboard"))
        elif "/admin/audit" in ref:
            return redirect(url_for("audit_list"))
    return redirect(url_for("alarm_list"))


@app.route("/admin/alarm/clear_all", methods=["POST"])
@login_required
def admin_alarm_clear_all():
    """一键清空（批量处理）所有待处理报警：将所有 Status='1' 的报警标记为已处理/误报。
    
    Returns:
        JSON 响应，包含处理结果状态码。
    """
    try:
        db = get_db()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.execute("UPDATE T_DetectResult SET Status='2', OperateUserId=?, OperateTime=?, UrgencyDegree='普通', OperateResult='误报无需处理', Description='一键清空/批量处理' WHERE Status='1'", (session["user_id"],))
        db.commit()
        add_log("报警事件", "批量清空", {"operator_id": session["user_id"]})
        return jsonify({"code": 200, "msg": "一键处理完成"})
    except Exception as e:
        logger.error(f"Clear all alarms error: {e}")
        return jsonify({"code": 500, "msg": str(e)}), 500


# --- 路由：事件审核 ---

@app.route("/admin/audit")
@login_required
def audit_list():
    """事件审核列表：仅超级管理员和审核人可访问，展示状态为'已处理/待审核'的报警记录。"""
    db = get_db()
    user = get_current_user()
    # 权限检查：只有超级管理员和审核人可以进入审核页面
    if user["RoleName"] not in ["超级管理员", "审核人"]:
        flash("您没有审核权限")
        return redirect(url_for("dashboard"))
    
    alarms = [dict(r) for r in db.execute(
        "SELECT dr.*, c.Name as CameraName, a.Name as AreaName, u.Name as OperatorName FROM T_DetectResult dr LEFT JOIN T_Camera c ON dr.CameraId=c.Id LEFT JOIN T_Area a ON dr.AreaId=a.Id LEFT JOIN T_User u ON dr.OperateUserId=u.Id WHERE dr.Status='2' ORDER BY dr.CreatTime DESC").fetchall()]
    return render_template_string(AUDIT_TEMPLATE, user=user, alarms=alarms)


@app.route("/admin/audit/approve/<int:aid>")
@login_required
def audit_approve(aid):
    """审核通过：将报警状态更新为 '3'（已审核），记录审核人和审核时间。
    
    Args:
        aid: 报警记录 ID
    """
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("UPDATE T_DetectResult SET Status='3', AuditUserId=?, AuditTime=? WHERE Id=?",
               (session["user_id"], now, aid))
    db.commit()
    add_log("事件审核", "审核通过", {"alarm_id": aid})
    flash("审核已通过")
    ref = request.referrer
    if ref:
        if "/dashboard" in ref:
            return redirect(url_for("dashboard"))
        elif "/admin/alarm" in ref:
            return redirect(url_for("alarm_list"))
    return redirect(url_for("audit_list"))


@app.route("/admin/audit/reject/<int:aid>")
@login_required
def audit_reject(aid):
    """审核驳回：将报警状态回退为 '1'（待处理），需要重新处理。
    
    Args:
        aid: 报警记录 ID
    """
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("UPDATE T_DetectResult SET Status='1', AuditUserId=?, AuditTime=? WHERE Id=?",
               (session["user_id"], now, aid))
    db.commit()
    add_log("事件审核", "审核驳回", {"alarm_id": aid})
    flash("已驳回，需重新处理")
    ref = request.referrer
    if ref:
        if "/dashboard" in ref:
            return redirect(url_for("dashboard"))
        elif "/admin/alarm" in ref:
            return redirect(url_for("alarm_list"))
    return redirect(url_for("audit_list"))


@app.route("/admin/simulate_error", methods=["POST"])
@login_required
def simulate_error():
    """模拟故障发生：往故障日志表手动写入一条模拟故障。"""
    try:
        data = request.get_json()
        error_type = data.get("type") # "device" or "camera"
        target_id = int(data.get("id", 1))
        error_code = data.get("code", "算力异常")
        error_msg = data.get("msg", "模拟异常测试日志")
        
        db = get_db()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if error_type == "device":
            dev = db.execute("SELECT * FROM T_Device WHERE Id=?", (target_id,)).fetchone()
            mac = dev["MAC"] if dev else "unknown"
            db.execute("INSERT INTO T_DeviceError (DeviceId, MAC, CreateTime, ErrorCode, ErrorMsg) VALUES (?,?,?,?,?)",
                       (target_id, mac, now_str, error_code, error_msg))
        else:
            cam = db.execute("SELECT * FROM T_Camera WHERE Id=?", (target_id,)).fetchone()
            mac = f"Port:{cam['WsPort']}" if cam else "unknown"
            db.execute("INSERT INTO T_CameraError (CameraId, MAC, CreateTime, ErrorCode, ErrorMsg) VALUES (?,?,?,?,?)",
                       (target_id, mac, now_str, error_code, error_msg))
        db.commit()
        return jsonify({"code": 200, "msg": "故障模拟成功，已记录至运维日志！"})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


# --- 路由：摄像头错误日志 ---

@app.route("/admin/camera_error")
@login_required
def camera_error_list():
    """摄像头错误日志列表页面：联表查询摄像头名称，按时间降序展示。"""
    db = get_db()
    check_and_log_faults(db)
    errors = [dict(r) for r in db.execute(
        "SELECT ce.*, c.Name as CameraName FROM T_CameraError ce LEFT JOIN T_Camera c ON ce.CameraId=c.Id ORDER BY ce.CreateTime DESC").fetchall()]
    return render_template_string(CAMERA_ERROR_TEMPLATE, user=get_current_user(), errors=errors)


# --- 路由：设备错误日志 ---

@app.route("/admin/device_error")
@login_required
def device_error_list():
    """设备错误日志列表页面：联表查询设备地址和 MAC，按时间降序展示。"""
    db = get_db()
    check_and_log_faults(db)
    errors = [dict(r) for r in db.execute(
        "SELECT de.*, d.Address as DeviceAddress, d.MAC as DeviceMAC FROM T_DeviceError de LEFT JOIN T_Device d ON de.DeviceId=d.Id ORDER BY de.CreateTime DESC").fetchall()]
    return render_template_string(DEVICE_ERROR_TEMPLATE, user=get_current_user(), errors=errors)


# --- 路由：系统日志 ---

@app.route("/admin/log/access")
@login_required
def access_log():
    """登录日志页面：展示最近 200 条用户登录记录（含用户名）。"""
    db = get_db()
    logs = [dict(r) for r in db.execute(
        "SELECT l.*, u.Name as UserName FROM T_UserLoginLog l LEFT JOIN T_User u ON l.UserId=u.Id ORDER BY l.LoginTime DESC LIMIT 200").fetchall()]
    return render_template_string(ACCESS_LOG_TEMPLATE, user=get_current_user(), logs=logs)


@app.route("/admin/log/operate")
@login_required
def operate_log():
    """操作日志页面：展示最近 200 条用户操作记录（含用户名）。"""
    db = get_db()
    logs = [dict(r) for r in db.execute(
        "SELECT l.*, u.Name as UserName FROM T_OperateLog l LEFT JOIN T_User u ON l.UserId=u.Id ORDER BY l.CreateTime DESC LIMIT 200").fetchall()]
    return render_template_string(OPERATE_LOG_TEMPLATE, user=get_current_user(), logs=logs)


# --- API 路由：边缘设备通信接口 ---

@app.route("/api/device/heartbeat", methods=["POST"])
def api_heartbeat():
    """设备心跳接口：接收边缘设备定期上报的状态信息。
    
    自动更新设备最后连接时间，同步摄像头位置和 WebSocket 端口信息，
    并返回当前系统配置（阈值、尺寸等）供设备使用。
    
    Returns:
        JSON: {"code": 200, "msg": "ok", "config": {...}} 或错误信息
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 400, "msg": "Invalid JSON"}), 400
        mac = data.get("device_mac", "")
        did = data.get("device_id", 1)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db = get_db()
        # 查找已注册设备：优先 MAC 匹配，其次 ID 匹配
        device = db.execute("SELECT * FROM T_Device WHERE MAC=? OR Id=?", (mac, did)).fetchone()
        if device:
            # 已存在设备，更新连接时间并清除自动错误标记
            db.execute("UPDATE T_Device SET LastConnectTime=?, AutoGenerateError='no' WHERE Id=?",
                       (now, device["Id"]))
        else:
            # 新设备自动注册
            auto_err = "yes" if data.get("status") != "online" else "no"
            db.execute("""INSERT INTO T_Device (MAC, ModelInfo, LastConnectTime, AutoGenerateError)
                VALUES (?,?,?,?)""", (mac, data.get("model_info", "YOLOv11"), now, auto_err))
        
        # 同步摄像头信息：如果心跳携带 camera_id，则更新或创建摄像头记录
        camera_id = data.get("camera_id")
        if camera_id:
            loc = data.get("location")
            port = data.get("websocket_port")
            lat = data.get("latitude")
            lng = data.get("longitude")
            camera = db.execute("SELECT * FROM T_Camera WHERE Id=?", (camera_id,)).fetchone()
            if camera:
                db.execute("UPDATE T_Camera SET Name=COALESCE(?, Name), WsPort=COALESCE(?, WsPort), Latitude=COALESCE(?, Latitude), Longitude=COALESCE(?, Longitude) WHERE Id=?", (loc, port, lat, lng, camera_id))
            else:
                db.execute("INSERT INTO T_Camera (Id, Name, WsPort, DeviceId, Latitude, Longitude) VALUES (?, ?, ?, ?, ?, ?)", (camera_id, loc, port, did, lat, lng))
        db.commit()
        # 读取系统配置并返回给设备
        site = db.execute("SELECT * FROM T_Site WHERE Id=1").fetchone()
        config = {"thresh": site["thresh"], "width": site["width"], "height": site["height"],
                  "video_times": site["video_times"], "heartBeat": site["heartBeat"], "exception_times": site["exception_times"]} if site else {}
        return jsonify({"code": 200, "msg": "ok", "config": config})
    except Exception as e:
        logger.error(f"Heartbeat error: {e}")
        return jsonify({"code": 500, "msg": str(e)}), 500


@app.route("/api/alarm", methods=["POST"])
def api_alarm():
    """火焰报警上报接口：接收边缘设备检测到的火焰报警数据。
    
    支持上传截图（picture）和录像（video）文件，解析检测结果中的最大置信度，
    存储到 T_DetectResult 表。
    
    Returns:
        JSON: {"code": 200, "msg": "ok", "alarm_id": id} 或错误信息
    """
    try:
        db = get_db()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 处理上传的截图文件
        picture_file = request.files.get("picture")
        video_file = request.files.get("video")
        picture_path = ""
        video_path = ""

        if picture_file:
            ext = os.path.splitext(picture_file.filename)[1] or ".jpg"
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}{ext}"
            filepath = UPLOAD_DIR / "pictures" / filename
            picture_file.save(str(filepath))
            picture_path = f"/uploads/pictures/{filename}"

        if video_file:
            ext = os.path.splitext(video_file.filename)[1] or ".mp4"
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}{ext}"
            filepath = UPLOAD_DIR / "videos" / filename
            video_file.save(str(filepath))
            video_path = f"/uploads/videos/{filename}"

        # 解析检测结果 JSON，提取最大置信度
        import json
        detections_json = request.form.get("detections", "[]")
        max_conf = 0.0
        try:
            detections = json.loads(detections_json)
            if detections:
                max_conf = max(float(d.get("confidence", 0.0)) for d in detections)
        except Exception:
            pass

        # 插入报警记录，状态默认为 '1'（待处理）
        desc = request.form.get("description", "")
        db.execute("""INSERT INTO T_DetectResult (Longitude, Latitude, Location, Picture, VideoUrl, AreaId, CreatTime, CameraId, DeviceId, Status, Confidence, Description)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (request.form.get("longitude", ""), request.form.get("latitude", ""),
                    request.form.get("location", ""), picture_path, video_path,
                    int(request.form.get("area_id", 1)), now,
                    int(request.form.get("camera_id", 1)), int(request.form.get("device_id", 1)), "1", max_conf, desc))
        db.commit()
        aid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.info(f"Alarm received: id={aid}")
        return jsonify({"code": 200, "msg": "ok", "alarm_id": aid})
    except Exception as e:
        logger.error(f"Alarm error: {e}")
        return jsonify({"code": 500, "msg": str(e)}), 500


@app.route("/api/device/error", methods=["POST"])
def api_device_error():
    """设备错误上报接口：接收边缘设备发送的错误信息，写入 T_DeviceError 表。
    
    Returns:
        JSON: {"code": 200, "msg": "ok"} 或错误信息
    """
    try:
        data = request.get_json()
        db = get_db()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.execute("INSERT INTO T_DeviceError (DeviceId, MAC, CreateTime, ErrorCode, ErrorMsg) VALUES (?,?,?,?,?)",
                   (data.get("device_id", 1), data.get("device_mac", ""), now, data.get("error_code", ""), data.get("error_msg", "")))
        db.commit()
        return jsonify({"code": 200, "msg": "ok"})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


@app.route("/api/camera/discover", methods=["POST"])
def api_camera_discover():
    """动态发现并注册摄像头接口：接收前端扫描发现的摄像头元数据并注册到 T_Camera 表中。
    
    Returns:
        JSON: {"code": 200, "msg": "ok"} 或错误信息
    """
    try:
        data = request.get_json() or {}
        camera_id = int(data.get("camera_id", 1))
        ws_port = int(data.get("ws_port", 9999))
        camera_ip = data.get("ip", "127.0.0.1")
        location = data.get("location", "未知位置")
        camera_name = data.get("camera_name", f"摄像头 {camera_id}")
        
        db = get_db()
        existing = db.execute("SELECT * FROM T_Camera WHERE Id = ?", (camera_id,)).fetchone()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if existing:
            db.execute("UPDATE T_Camera SET WsPort = ?, IP = ?, CameraUrl = 'ws://'||?||':'||?, Name = ?, InstallTime = ? WHERE Id = ?",
                       (ws_port, camera_ip, camera_ip, ws_port, camera_name, now, camera_id))
        else:
            lat_offset = (camera_id % 3) * 0.001
            lng_offset = (camera_id % 2) * 0.001
            lat = f"{29.563009 + lat_offset:.6f}"
            lng = f"{106.551556 + lng_offset:.6f}"
            db.execute("""INSERT INTO T_Camera (Id, IP, MAC, CameraUrl, Name, Longitude, Latitude, AreaId, Type, DeviceId, InstallTime, WsPort)
                          VALUES (?, ?, ?, 'ws://'||?||':'||?, ?, ?, ?, 1, '发现设备', 1, ?, ?)""",
                       (camera_id, camera_ip, f"DISC:CAM:{camera_id}", camera_ip, ws_port, camera_name, lng, lat, now, ws_port))
        
        db.commit()
        return jsonify({"code": 200, "msg": "ok"})
    except Exception as e:
        logger.error(f"Discover camera error: {e}")
        return jsonify({"code": 500, "msg": str(e)}), 500


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """提供上传文件的静态访问，如截图、录像、Logo 等。
    支持 HTTP Range 请求以确保视频在 Chrome/Safari 中能自由拖动和播放。
    """
    import re
    from flask import Response, abort
    
    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        return abort(404)

    # 针对非视频文件，直接使用 send_from_directory
    if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        return send_from_directory(str(UPLOAD_DIR), filename)

    # 提供 206 Partial Content Range 拖动支持
    file_size = filepath.stat().st_size
    range_header = request.headers.get("Range", None)
    if not range_header:
        return send_from_directory(str(UPLOAD_DIR), filename)

    byte1, byte2 = 0, None
    m = re.search(r"bytes=(\d+)-(\d*)", range_header)
    if m:
        g = m.groups()
        if g[0]:
            byte1 = int(g[0])
        if g[1]:
            byte2 = int(g[1])

    length = file_size - byte1
    if byte2 is not None and byte2 != "":
        try:
            byte2_val = int(byte2)
            length = byte2_val - byte1 + 1
        except ValueError:
            pass

    data = None
    with open(filepath, "rb") as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
    rv.headers.add("Content-Range", f"bytes {byte1}-{byte1 + len(data) - 1}/{file_size}")
    rv.headers.add("Accept-Ranges", "bytes")
    return rv


@app.route("/api/stats")
@login_required
def api_stats():
    """数据统计 API：返回大屏仪表盘所需的各类统计数据（JSON 格式）。
    
    包含区域分布、时间趋势、真假漏报统计、近期报警列表、月度排行等。
    
    Returns:
        JSON: 包含所有统计数据的字典
    """
    db = get_db()
    check_and_log_faults(db)
    # 各时间段统计数据
    today_start = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S")

    # 各区域报警数量分布（用于地图热力图/柱状图）
    area = [dict(r) for r in db.execute(
        "SELECT a.Name as name, COUNT(dr.Id) as value FROM T_Area a LEFT JOIN T_DetectResult dr ON a.Id=dr.AreaId GROUP BY a.Id").fetchall()]
    
    # 近30天每日报警趋势
    time_total = [dict(r) for r in db.execute(
        "SELECT strftime('%Y-%m-%d', CreatTime) as date, COUNT(*) as count FROM T_DetectResult WHERE CreatTime > ? GROUP BY date", (month_ago,)).fetchall()]
    time_true = [dict(r) for r in db.execute(
        "SELECT strftime('%Y-%m-%d', CreatTime) as date, COUNT(*) as count FROM T_DetectResult WHERE CreatTime > ? AND Status='2' AND OperateResult='火灾已确认并报警' GROUP BY date", (month_ago,)).fetchall()]
    time_false = [dict(r) for r in db.execute(
        "SELECT strftime('%Y-%m-%d', CreatTime) as date, COUNT(*) as count FROM T_DetectResult WHERE CreatTime > ? AND Status='2' AND OperateResult='误报无需处理' GROUP BY date", (month_ago,)).fetchall()]
    time_missed = [dict(r) for r in db.execute(
        "SELECT strftime('%Y-%m-%d', CreatTime) as date, COUNT(*) as count FROM T_DetectResult WHERE CreatTime > ? AND Status='2' AND OperateResult='漏报记录' GROUP BY date", (month_ago,)).fetchall()]
    time_pending = [dict(r) for r in db.execute(
        "SELECT strftime('%Y-%m-%d', CreatTime) as date, COUNT(*) as count FROM T_DetectResult WHERE CreatTime > ? AND Status='1' GROUP BY date", (month_ago,)).fetchall()]

    dates = []
    for i in range(29, -1, -1):
        dates.append((datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"))

    def get_counts_by_date(raw):
        res = {}
        for r in raw:
            res[r["date"]] = r["count"]
        return res

    total_map = get_counts_by_date(time_total)
    true_map = get_counts_by_date(time_true)
    false_map = get_counts_by_date(time_false)
    missed_map = get_counts_by_date(time_missed)
    pending_map = get_counts_by_date(time_pending)

    time_data_padded = []
    for d in dates:
        time_data_padded.append({
            "date": d,
            "total": total_map.get(d, 0),
            "true": true_map.get(d, 0),
            "false": false_map.get(d, 0),
            "missed": missed_map.get(d, 0),
            "pending": pending_map.get(d, 0)
        })

    total = db.execute("SELECT COUNT(*) as c FROM T_DetectResult").fetchone()["c"]
    pending_alarms = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE Status='1'").fetchone()["c"]

    today_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE CreatTime >= ?", (today_start,)).fetchone()["c"]
    week_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE CreatTime > ?", (week_ago,)).fetchone()["c"]
    month_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE CreatTime > ?", (month_ago,)).fetchone()["c"]
    year_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE CreatTime > ?", (year_ago,)).fetchone()["c"]

    # 真火警 / 误报 / 漏报 分类统计
    true_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE OperateResult='火灾已确认并报警'").fetchone()["c"]
    false_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE OperateResult='误报无需处理'").fetchone()["c"]
    missed_count = db.execute("SELECT COUNT(*) as c FROM T_DetectResult WHERE OperateResult='漏报记录'").fetchone()["c"]

    # 设备在线/离线统计
    total_devices = db.execute("SELECT COUNT(*) as c FROM T_Device").fetchone()["c"]
    online_threshold = (datetime.now() - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")
    online_count = db.execute("SELECT COUNT(*) as c FROM T_Device WHERE LastConnectTime >= ?", (online_threshold,)).fetchone()["c"]
    offline_count = max(0, total_devices - online_count)

    # AI处理率 (已处理数 / 总数 * 100)
    ai_rate = round(((total - pending_alarms) / total * 100) if total > 0 else 100, 1)

    earliest_row = db.execute("SELECT MIN(CreatTime) as m FROM T_DetectResult").fetchone()
    earliest_time = earliest_row["m"] if earliest_row and earliest_row["m"] else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 月度区域报警排名（Top 5）
    monthly_ranking = [dict(r) for r in db.execute(
        "SELECT COALESCE(NULLIF(dr.Location, ''), '未知区域') as name, COUNT(*) as count FROM T_DetectResult dr WHERE dr.CreatTime > ? GROUP BY dr.Location ORDER BY count DESC LIMIT 5", (month_ago,)).fetchall()]
    max_rank = max([r["count"] for r in monthly_ranking]) if monthly_ranking else 1

    # 查询所有摄像头数据用于大屏幕实时地图
    cameras = [dict(r) for r in db.execute("SELECT * FROM T_Camera ORDER BY Id").fetchall()]

    recent_alarms = [dict(r) for r in db.execute(
        "SELECT dr.*, c.Name as CameraName, a.Name as AreaName, u.Name as OperatorName FROM T_DetectResult dr LEFT JOIN T_Camera c ON dr.CameraId=c.Id LEFT JOIN T_Area a ON dr.AreaId=a.Id LEFT JOIN T_User u ON dr.OperateUserId=u.Id ORDER BY (CASE WHEN dr.Status='1' THEN 1 WHEN dr.OperateResult='误报无需处理' THEN 4 WHEN dr.OperateResult='漏报记录' THEN 3 ELSE 2 END) ASC, dr.Confidence DESC, dr.CreatTime DESC LIMIT 30").fetchall()]

    return jsonify({
        "area_stats": area,
        "time_stats": time_data_padded,
        "total": total,
        "today_count": today_count,
        "week_count": week_count,
        "month_count": month_count,
        "year_count": year_count,
        "true_count": true_count,
        "false_count": false_count,
        "missed_count": missed_count,
        "pending_alarms": pending_alarms,
        "online_count": online_count,
        "offline_count": offline_count,
        "ai_rate": ai_rate,
        "recent_alarms": recent_alarms,
        "monthly_ranking": monthly_ranking,
        "max_rank": max_rank,
        "earliest_time": earliest_time,
        "cameras": cameras
    })


# --- HTML 模板定义 ---

# 登录页面模板
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>视频AI智能识别及预警管理系统 - 登录</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
.glass-panel {
  background: rgba(15, 23, 42, 0.45);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.06);
}
</style>
</head>
<body class="bg-gradient-to-tr from-[#030712] via-[#091124] to-[#030712] text-slate-100 min-h-screen flex items-center justify-center font-sans p-4">
<div class="glass-panel w-full max-w-md rounded-2xl p-8 shadow-[0_20px_50px_rgba(0,0,0,0.5)] flex flex-col gap-6 relative overflow-hidden">
    <!-- Decorative glow -->
    <div class="absolute -right-10 -top-10 w-32 h-32 bg-cyan-500/10 rounded-full blur-2xl"></div>
    <div class="absolute -left-10 -bottom-10 w-32 h-32 bg-rose-500/10 rounded-full blur-2xl"></div>
    
    <div class="flex flex-col items-center gap-2 text-center">
        <div class="w-14 h-14 bg-gradient-to-br from-rose-500 to-orange-500 rounded-2xl flex items-center justify-center shadow-lg shadow-rose-950/40">
            <span class="text-3xl">🔥</span>
        </div>
        <h2 class="text-lg font-bold tracking-wider text-slate-100 mt-2">视频AI智能识别及预警管理</h2>
        <p class="text-xs text-slate-400">后台管理平台登录</p>
    </div>
    
    {% with msgs = get_flashed_messages() %}
    {% if msgs %}
    <div class="bg-rose-500/10 border border-rose-500/20 text-rose-450 px-4 py-2.5 rounded-lg text-xs font-semibold animate-pulse">
        {{ msgs[0] }}
    </div>
    {% endif %}
    {% endwith %}
    
    <form method="post" class="flex flex-col gap-4 text-xs">
        <div class="flex flex-col gap-1.5">
            <label class="text-slate-400 font-medium">账号</label>
            <input type="text" name="account" placeholder="请输入您的账号" required autofocus class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3.5 py-2.5 text-slate-200 placeholder-slate-650 focus:outline-none focus:border-cyan-500/50 transition">
        </div>
        <div class="flex flex-col gap-1.5">
            <label class="text-slate-400 font-medium">密码</label>
            <input type="password" name="password" placeholder="请输入您的密码" required class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3.5 py-2.5 text-slate-200 placeholder-slate-650 focus:outline-none focus:border-cyan-500/50 transition">
        </div>
        <button type="submit" class="w-full bg-gradient-to-r from-rose-600 to-orange-600 hover:from-rose-500 hover:to-orange-500 text-white font-bold py-3 rounded-lg mt-2 transition duration-300 active:scale-95 shadow-lg shadow-rose-950/30">登 录</button>
    </form>
    
    <div class="border-t border-slate-800/80 pt-4 flex flex-col gap-1 text-center text-[10px] text-slate-500">
        <p>测试账号: 管理员 (admin/123456) | 处理人 (chuli001/123456)</p>
        <p class="font-mono mt-0.5">&copy; 2026 Fire AI Detection System. All rights reserved.</p>
    </div>
</div>
</body>
</html>
"""

# 管理后台公共导航栏组件模板（固定顶部）
BASE_NAV = """
<header class="flex justify-between items-center h-14 border-b border-slate-800/60 bg-[#050c18]/90 backdrop-blur-md px-6 z-[1000] w-full fixed top-0 left-0 right-0 text-slate-100 font-sans">
  <div class="flex items-center gap-3">
    <svg class="w-6 h-6 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
      <line x1="8" y1="21" x2="16" y2="21"></line>
      <line x1="12" y1="17" x2="12" y2="21"></line>
    </svg>
    <h1 class="m-0 font-bold text-base tracking-wide text-white">
      火焰智能检测预警平台
    </h1>
  </div>
  <nav class="flex items-center gap-1 text-sm">
    <a href="/dashboard" id="nav-dashboard" class="relative flex items-center gap-1.5 px-4 py-2 font-semibold transition-all duration-300 text-slate-300 hover:text-white">
      数据大屏
      <span id="nav-dash-line" class="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-cyan-400 rounded-full shadow-[0_0_8px_#22d3ee] hidden"></span>
    </a>
    {% if user.RoleName == '超级管理员' %}
    <a href="/admin/device" id="nav-device" class="relative flex items-center gap-1.5 px-4 py-2 font-semibold transition-all duration-300 text-slate-400 hover:text-white">
      资源管理
      <span id="nav-device-line" class="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-cyan-400 rounded-full shadow-[0_0_8px_#22d3ee] hidden"></span>
    </a>
    <a href="/admin/config" id="nav-config" class="relative flex items-center gap-1.5 px-4 py-2 font-semibold transition-all duration-300 text-slate-400 hover:text-white">
      系统设置
      <span id="nav-config-line" class="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-cyan-400 rounded-full shadow-[0_0_8px_#22d3ee] hidden"></span>
    </a>
    {% endif %}
    <a href="/admin/alarm" id="nav-alarm" class="relative flex items-center gap-1.5 px-4 py-2 font-semibold transition-all duration-300 text-slate-400 hover:text-white">
      预警中心
      <span id="nav-alarm-line" class="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-cyan-400 rounded-full shadow-[0_0_8px_#22d3ee] hidden"></span>
    </a>
  </nav>
  <div class="flex items-center gap-5 text-sm">
    <span id="nav-clock" class="text-cyan-300 font-mono font-bold tracking-widest mr-1 hidden md:inline-block text-base drop-shadow-[0_0_6px_rgba(34,211,238,0.4)]"></span>
    <span class="inline-flex items-center gap-1.5 text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full font-semibold text-xs">
      <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_4px_#34d399]"></span> 在线
    </span>
    <span class="text-slate-300 font-medium">{{ user.Name }}</span>
    <a href="/logout" class="text-rose-400 hover:text-rose-300 transition-colors font-semibold">退出</a>
  </div>
</header>
<div class="h-14"></div>
<script>
document.addEventListener("DOMContentLoaded", function() {
  function updateNavClock() {
    const el = document.getElementById("nav-clock");
    if (el) {
      const now = new Date();
      el.textContent = now.toLocaleTimeString('zh-CN', { hour12: false });
    }
  }
  updateNavClock();
  setInterval(updateNavClock, 1000);

  const path = window.location.pathname;
  let activeId = "";
  if (path.includes("/dashboard")) activeId = "nav-dashboard";
  else if (path.includes("/admin/device") || path.includes("/admin/camera")) activeId = "nav-device";
  else if (path.includes("/admin/config") || path.includes("/admin/branch") || path.includes("/admin/user") || path.includes("/admin/role") || path.includes("/admin/dictionary")) activeId = "nav-config";
  else if (path.includes("/admin/alarm") || path.includes("/admin/audit") || path.includes("/admin/camera_error") || path.includes("/admin/device_error") || path.includes("/admin/log")) activeId = "nav-alarm";

  if (activeId) {
    const activeEl = document.getElementById(activeId);
    if (activeEl) {
      activeEl.classList.remove("text-slate-400");
      activeEl.classList.add("text-white");
      const line = document.getElementById(activeId + "-line");
      if (line) line.classList.remove("hidden");
    }
  }
});
</script>
"""

# 数据大屏仪表盘模板（ECharts + 百度地图）
DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>视频 AI 智能识别及预警平台</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
  <link rel="stylesheet" href="https://cdn.bootcdn.net/ajax/libs/leaflet/1.9.4/leaflet.css" />
  <script src="https://cdn.bootcdn.net/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
  <style>
    .tech-grid-diagonal {
      background-image: 
      	linear-gradient(rgba(34, 211, 238, 0.06) 1px, transparent 1px),
      	linear-gradient(90deg, rgba(34, 211, 238, 0.06) 1px, transparent 1px),
      	repeating-linear-gradient(45deg, rgba(244, 63, 94, 0.02) 0px, rgba(244, 63, 94, 0.02) 10px, transparent 10px, transparent 20px);
      background-size: 20px 20px, 20px 20px, 15px 15px;
    }
    .scrollbar-thin::-webkit-scrollbar { width: 4px; height: 4px; }
    .scrollbar-thin::-webkit-scrollbar-track { background: transparent; }
    .scrollbar-thin::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 2px; }
    .scrollbar-thin::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.3); }
    
    /* Smooth modern glassmorphism panels with transition effects */
    .glass-panel {
      background: rgba(15, 23, 42, 0.45);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.06);
      transition: border-color 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease, transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .glass-panel:hover {
      border-color: rgba(34, 211, 238, 0.2);
      box-shadow: 0 10px 30px -10px rgba(6, 182, 212, 0.12), 0 1px 1px rgba(255, 255, 255, 0.05) inset;
      background: rgba(15, 23, 42, 0.55);
    }
    
    /* Micro-scale animations for hover states */
    .hover-scale {
      transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.2s ease, filter 0.2s ease;
    }
    .hover-scale:hover {
      transform: scale(1.02);
      filter: brightness(1.1);
    }
    .hover-scale:active {
      transform: scale(0.98);
    }
    
    /* Slide in up animation for timeline entries and charts */
    @keyframes slideInUp {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .timeline-item-anim {
      animation: slideInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    /* Scanning effect for stream screen */
    @keyframes scanning {
      0% { top: 0%; opacity: 0.1; }
      50% { opacity: 0.8; }
      100% { top: 100%; opacity: 0.1; }
    }
    .scan-line {
      position: absolute;
      width: 100%;
      height: 3px;
      background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.7), transparent);
      animation: scanning 3s infinite linear;
      pointer-events: none;
    }

    /* Pulse effect for status points */
    @keyframes radarPulse {
      0% { transform: scale(0.9); opacity: 0.8; }
      50% { opacity: 0.4; }
      100% { transform: scale(1.6); opacity: 0; }
    }
    .radar-pulse {
      position: absolute;
      width: 100%;
      height: 100%;
      border-radius: 50%;
      background-color: currentColor;
      animation: radarPulse 2s infinite ease-out;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes scaleIn {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }
    .animate-fade-in {
      animation: fadeIn 0.2s ease-out forwards;
    }
    .animate-scale-in {
      animation: scaleIn 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }
    
    /* Custom Leaflet Map Styles to fit Dark Glass Panel theme */
    .leaflet-container {
      background: #030712 !important;
      font-family: inherit !important;
    }
    .leaflet-popup-content-wrapper {
      background: rgba(15, 23, 42, 0.92) !important;
      backdrop-filter: blur(12px) !important;
      -webkit-backdrop-filter: blur(12px) !important;
      border: 1px solid rgba(255, 255, 255, 0.1) !important;
      color: #f1f5f9 !important;
      border-radius: 14px !important;
      box-shadow: 0 15px 35px rgba(0,0,0,0.6) !important;
      padding: 4px !important;
    }
    .leaflet-popup-content {
      margin: 8px 12px !important;
      line-height: 1.4 !important;
    }
    .leaflet-popup-tip {
      background: rgba(15, 23, 42, 0.92) !important;
      border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .leaflet-bar {
      border: 1px solid rgba(255, 255, 255, 0.08) !important;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
      border-radius: 8px !important;
      overflow: hidden;
    }
    .leaflet-bar a {
      background: rgba(15, 23, 42, 0.85) !important;
      color: #22d3ee !important;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
      transition: background-color 0.2s;
    }
    .leaflet-bar a:hover {
      background: rgba(34, 211, 238, 0.15) !important;
      color: #22d3ee !important;
    }
    .custom-map-icon {
      background: none !important;
      border: none !important;
    }
  </style>
</head>
<body class="bg-gradient-to-tr from-[#030712] via-[#091124] to-[#030712] text-slate-100 h-screen w-screen overflow-hidden flex flex-col font-sans">
""" + BASE_NAV + """
<main class="flex-1 grid grid-cols-12 gap-4 p-4 h-[calc(100vh-56px)] overflow-hidden">
<!-- Left Column: 实时预警记录 -->
<section class="col-span-3 flex flex-col h-full overflow-hidden">
  <div class="flex-1 h-full glass-panel rounded-2xl p-4 shadow-[0_8px_32px_rgba(0,0,0,0.37)] flex flex-col gap-3 overflow-hidden border border-slate-800/80">
    <div class="flex flex-col gap-2.5 shrink-0">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="w-1.5 h-3.5 bg-rose-500 rounded-full shadow-[0_0_8px_#f43f5e]"></span>
          <span class="text-sm font-bold tracking-wider text-slate-200">实时预警记录</span>
          <span class="text-rose-400 text-xs font-bold bg-rose-500/10 border border-rose-500/20 px-2.5 py-0.5 rounded-full" id="pendingAlarmsCount">{{ pending_alarms }}</span>
        </div>
      </div>
      <div class="relative">
        <input type="text" id="searchLocation" placeholder="检索位置/设备..." class="w-full bg-slate-950/50 border border-slate-800/80 rounded-lg pl-8 pr-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 transition">
        <svg class="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
        <span class="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
          <input type="datetime-local" id="startTime" class="w-0 opacity-0 absolute" style="width:0;height:0;border:0;padding:0;margin:0;">
          <button onclick="document.getElementById('startTime').showPicker()" class="text-slate-500 hover:text-cyan-400 transition text-xs" title="开始时间">📅</button>
        </span>
      </div>
      <div class="flex flex-wrap gap-1.5 text-xs">
        <input type="datetime-local" id="endTime" class="w-0 opacity-0 absolute" style="width:0;height:0;border:0;padding:0;margin:0;">
        <button onclick="document.getElementById('endTime').showPicker()" class="flex items-center gap-1 bg-slate-950/50 border border-slate-800/80 rounded px-2 py-0.5 text-slate-400 hover:text-cyan-400 transition" title="结束时间">📅 结束</button>
        <button id="filterBtnAll" onclick="filterByTimeRange('all')" class="bg-slate-950/50 border border-slate-800/80 rounded px-2 py-0.5 text-slate-400 hover:text-cyan-400 hover:border-cyan-500/30 transition">全部</button>
        <button id="filterBtnToday" onclick="filterByTimeRange('today')" class="bg-slate-950/50 border border-slate-800/80 rounded px-2 py-0.5 text-slate-400 hover:text-cyan-400 hover:border-cyan-500/30 transition">今日</button>
        <button id="filterBtnWeek" onclick="filterByTimeRange('week')" class="bg-slate-950/50 border border-slate-800/80 rounded px-2 py-0.5 text-slate-400 hover:text-orange-400 hover:border-orange-500/30 transition">本周</button>
        <button id="filterBtnPending" onclick="filterByTimeRange('pending')" class="bg-slate-950/50 border border-slate-850/80 rounded px-2 py-0.5 text-slate-400 hover:text-rose-400 hover:border-rose-500/30 transition">待处理</button>
        <button id="filterBtnProcessed" onclick="filterByTimeRange('processed')" class="bg-slate-950/50 border border-slate-800/80 rounded px-2 py-0.5 text-slate-400 hover:text-blue-400 hover:border-blue-500/30 transition">已处理</button>
        <button id="filterBtnFalse" onclick="filterByTimeRange('false_alarm')" class="bg-slate-950/50 border border-slate-800/80 rounded px-2 py-0.5 text-slate-400 hover:text-emerald-400 hover:border-emerald-500/30 transition">误报</button>
        <button id="filterBtnMissed" onclick="filterByTimeRange('missed')" class="bg-slate-950/50 border border-slate-800/80 rounded px-2 py-0.5 text-slate-400 hover:text-amber-400 hover:border-amber-500/30 transition">漏报</button>
      </div>
    </div>
    <div class="flex-1 overflow-y-auto pr-1 flex flex-col gap-2 text-sm scrollbar-thin" id="timelineContainer">
      <div class="relative border-l border-slate-800 ml-2 pl-4 flex flex-col gap-3 py-2" id="timelineList">
        {% for a in recent_alarms %}
        {% set is_smoke = a.Description and '烟雾' in a.Description and '火焰' not in a.Description %}
        <div class="relative mb-2 timeline-item-anim">
          <span class="absolute -left-[21px] mt-1 w-2.5 h-2.5 rounded-full border bg-[#050c18] flex items-center justify-center
            {% if a.Status == '1' %}
              {% if is_smoke %} border-sky-500 shadow-[0_0_6px_#0ea5e9]
              {% else %} border-rose-500 shadow-[0_0_6px_#f43f5e]
              {% endif %}
            {% elif a.OperateResult == '误报无需处理' %} border-emerald-500 shadow-[0_0_6px_#10b981]
            {% elif a.OperateResult == '漏报记录' %} border-amber-500 shadow-[0_0_6px_#f59e0b]
            {% else %} border-blue-500 shadow-[0_0_6px_#3b82f6]
            {% endif %}">
            <span class="w-1.5 h-1.5 rounded-full relative flex
              {% if a.Status == '1' %}
                {% if is_smoke %} text-sky-500 bg-sky-500
                {% else %} text-rose-500 bg-rose-500
                {% endif %}
              {% elif a.OperateResult == '误报无需处理' %} text-emerald-500 bg-emerald-500
              {% elif a.OperateResult == '漏报记录' %} text-amber-500 bg-amber-500
              {% else %} text-blue-500 bg-blue-500
              {% endif %}">
              {% if a.Status == '1' or a.OperateResult == '漏报记录' %}
              <span class="radar-pulse"></span>
              {% endif %}
            </span>
          </span>
          <div class="glass-panel rounded-xl p-3 flex flex-col gap-1.5 cursor-pointer hover:border-cyan-500/20 hover:shadow-[0_0_15px_rgba(6,182,212,0.15)] transition duration-300 border-l-[3px]
            {% if a.Status == '1' %}
              {% if is_smoke %} border-l-sky-500 shadow-[0_0_8px_rgba(14,165,233,0.1)]
              {% else %} border-l-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.1)]
              {% endif %}
            {% elif a.OperateResult == '误报无需处理' %} border-l-emerald-500
            {% elif a.OperateResult == '漏报记录' %} border-l-amber-500
            {% else %} border-l-blue-500
            {% endif %}" onclick="showAlarmDetail({{ a.Id }})">
            <div class="flex justify-between items-center">
              <span class="{% if is_smoke %}text-sky-400{% else %}text-rose-400{% endif %} font-semibold text-xs">{% if is_smoke %}烟雾报警{% else %}火焰预警{% endif %} {% if a.Confidence %}({{ (a.Confidence*100)|round(1) }}%){% endif %}</span>
              {% if a.Status == '1' %}
                {% if is_smoke %}
                <span class="text-[10px] bg-sky-500/10 text-sky-400 border border-sky-500/20 px-1.5 py-0.5 rounded font-bold">待处理</span>
                {% else %}
                <span class="text-[10px] bg-rose-500/10 text-rose-400 border border-rose-500/20 px-1.5 py-0.5 rounded font-bold">待处理</span>
                {% endif %}
              {% elif a.OperateResult == '误报无需处理' %}
              <span class="text-[10px] bg-emerald-500/15 text-emerald-450 border border-emerald-500/35 px-1.5 py-0.5 rounded font-bold">排除误报</span>
              {% elif a.OperateResult == '漏报记录' %}
              <span class="text-[10px] bg-amber-500/15 text-amber-400 border border-amber-500/35 px-1.5 py-0.5 rounded font-bold animate-pulse">漏报记录</span>
              {% else %}
              <span class="text-[10px] bg-blue-500/15 text-blue-400 border border-blue-500/35 px-1.5 py-0.5 rounded font-bold">已处理</span>
              {% endif %}
            </div>
            <div class="flex justify-between items-center text-xs text-slate-350 mt-1">
              <span class="truncate max-w-[120px]">{{ a.Location or a.AreaName or '未知位置' }}</span>
              <span class="text-cyan-400 font-mono text-[10px] truncate max-w-[80px]">{{ a.CameraName or '摄像头1' }}</span>
            </div>
            <div class="flex justify-between items-center text-[10px] text-slate-500 mt-0.5">
              <span>{{ a.CreatTime or '--' }}</span>
              <span class="text-cyan-500/70 hover:text-cyan-400 transition">查看详情 →</span>
            </div>
          </div>
        </div>
        {% else %}
        <div class="flex items-center justify-center h-full text-slate-600 text-sm py-8">暂无报警记录</div>
        {% endfor %}
      </div>
    </div>
  </div>
</section>

<!-- Middle Column: 中央监控终端 + AI 智能抓拍 -->
<section class="col-span-6 flex flex-col gap-4 h-full overflow-hidden">
  <!-- 中央监控终端 -->
  <div class="flex-1 glass-panel rounded-2xl p-4 shadow-[0_8px_32px_rgba(0,0,0,0.37)] flex flex-col gap-3 overflow-hidden">
    <div class="flex items-center justify-between shrink-0">
      <div class="flex items-center gap-3">
        <h2 class="text-sm font-bold tracking-wider text-slate-200 flex items-center gap-2">
          <span class="w-1.5 h-3.5 bg-cyan-400 rounded-full shadow-[0_0_8px_#22d3ee]"></span>
          中央监控终端
        </h2>
        <div class="flex bg-slate-950/80 rounded-lg p-0.5 border border-slate-800/80">
          <button id="btnViewVideo" onclick="switchCenterView('video')" class="px-2.5 py-1 rounded-md text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-sm transition active:scale-95">📹 实时监控</button>
          <button id="btnViewMap" onclick="switchCenterView('map')" class="px-2.5 py-1 rounded-md text-[10px] font-semibold text-slate-400 hover:text-slate-200 transition active:scale-95">🗺️ 预警地图</button>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <button onclick="openScanModal()" class="text-xs bg-slate-800/50 hover:bg-slate-800 border border-slate-700 text-slate-300 px-2.5 py-1 rounded-lg transition active:scale-95 font-semibold flex items-center gap-1">⚙️ 扫描设置</button>
        <select id="cameraSelector" onchange="changeCameraStream()" class="bg-slate-950/50 border border-slate-800 rounded px-2.5 py-1 text-slate-300 focus:outline-none focus:border-cyan-500/50 transition text-xs font-semibold min-w-[140px]">
          <option value="" disabled selected>🔍 正在检索监控...</option>
        </select>
        <span class="text-xs text-slate-500 bg-slate-950/50 px-3 py-1 rounded-full border border-slate-800">1280x720 | WS</span>
      </div>
    </div>
    <div class="flex-1 bg-[#030712] rounded-xl relative overflow-hidden border border-slate-900 shadow-inner min-h-0 flex">
      <!-- 实时监控画面 -->
      <div id="videoFeedContainer" class="w-full h-full relative flex items-center justify-center">
        <!-- Tech corner borders -->
        <div class="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-cyan-500/30 rounded-tl-lg z-20 pointer-events-none"></div>
        <div class="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-cyan-500/30 rounded-tr-lg z-20 pointer-events-none"></div>
        <div class="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-cyan-500/30 rounded-bl-lg z-20 pointer-events-none"></div>
        <div class="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-cyan-500/30 rounded-br-lg z-20 pointer-events-none"></div>
        <!-- Green dot grid background -->
        <div class="tech-grid-diagonal absolute inset-0 opacity-60"></div>
        <div class="scan-line"></div>
        <img id="cameraFrame" class="hidden absolute inset-0 w-full h-full object-contain z-10">
        <div id="videoOffline" class="flex flex-col items-center gap-4 z-10 text-center p-6 bg-slate-950/40 rounded-2xl border border-slate-900/60 shadow-xl max-w-sm">
          <div class="relative flex items-center justify-center w-16 h-16">
            <div class="absolute inset-0 rounded-full border-2 border-dashed border-cyan-500/20 animate-spin" style="animation-duration:3s"></div>
            <svg class="w-8 h-8 text-cyan-400/70" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <polygon points="5 3 19 12 5 21 5 3" fill="rgba(34,211,238,0.1)"/>
            </svg>
          </div>
          <div class="flex flex-col gap-1">
            <span class="text-cyan-400 text-sm font-bold tracking-widest uppercase">信号连接中断</span>
            <span class="text-slate-500 text-xs">AWAITING VIDEO STREAM...</span>
          </div>
        </div>
        <span id="videoTag" class="hidden absolute top-3 left-3 flex items-center gap-1.5 z-20">
          <span class="w-2 h-2 rounded-full bg-rose-500 animate-pulse shadow-[0_0_6px_#f43f5e]"></span>
          <span class="text-rose-400 text-xs font-bold tracking-wider">REC CH-01 MAIN FEED</span>
          <span class="text-slate-500 text-[10px] ml-1">1080P/60FPS</span>
        </span>
      </div>
      <!-- 地图容器 -->
      <div id="mapContainer" class="hidden w-full h-full relative rounded-xl"></div>
    </div>
  </div>

  <!-- AI 智能抓拍 -->
  <div class="h-[235px] shrink-0 glass-panel rounded-2xl p-4 shadow-[0_8px_32px_rgba(0,0,0,0.37)] flex flex-col gap-2.5 overflow-hidden border border-slate-800/80">
    <h2 class="text-sm font-bold tracking-wider text-slate-200 flex items-center gap-2 shrink-0">
      <span class="w-1.5 h-3.5 bg-orange-500 rounded-full shadow-[0_0_8px_#f97316]"></span>
      AI 智能抓拍
    </h2>
    <div class="flex-1 flex gap-3 pb-1 overflow-x-auto scrollbar-thin" id="snapshotsContainer">
      {% set snapshots = recent_alarms|selectattr('Picture')|list %}
      {% for i in range(5) %}
        {% if i < snapshots|length %}
          {% set a = snapshots[i] %}
          {% set is_smoke = a.Description and '烟雾' in a.Description and '火焰' not in a.Description %}
          <div class="w-[calc(20%-10px)] min-w-[160px] shrink-0 rounded-xl bg-slate-950/40 border {% if is_smoke %}border-sky-500/50 shadow-[0_0_12px_rgba(14,165,233,0.15)]{% else %}border-rose-500/50 shadow-[0_0_12px_rgba(244,63,94,0.15)]{% endif %} p-2 flex flex-col gap-1.5 cursor-pointer hover:border-cyan-500/30 transition duration-300" onclick="showAlarmDetail({{ a.Id }})">
            <div class="relative aspect-video rounded-lg overflow-hidden border border-slate-800/60">
              <img src="{{ a.Picture }}" class="w-full h-full object-cover">
              {% if is_smoke %}
              <span class="absolute top-1 left-1 bg-sky-600/90 text-white text-[9px] font-bold px-1.5 py-0.5 rounded tracking-wider">烟雾报警</span>
              <span class="absolute bottom-1 right-1 bg-sky-600/80 text-white text-[9px] font-bold px-1.5 py-0.5 rounded">SMOKE {{ ((a.Confidence or 0.95)*100)|round|int }}%</span>
              {% else %}
              <span class="absolute top-1 left-1 bg-rose-600/90 text-white text-[9px] font-bold px-1.5 py-0.5 rounded tracking-wider">火灾预警</span>
              <span class="absolute bottom-1 right-1 bg-pink-600/80 text-white text-[9px] font-bold px-1.5 py-0.5 rounded">FIRE {{ ((a.Confidence or 0.95)*100)|round|int }}%</span>
              {% endif %}
            </div>
            <div class="flex justify-between items-center text-[10px] mt-0.5">
              <span class="text-orange-400 font-mono font-semibold">{{ a.CreatTime[11:19] if a.CreatTime else '--' }}</span>
              <span class="text-slate-400 truncate max-w-[80px] font-medium">{{ a.Location or '--' }}</span>
            </div>
            <div class="flex justify-between items-center mt-1">
              <span class="text-[9px] text-slate-500">通道-0{{ a.CameraId or 1 }}</span>
              {% if is_smoke %}
              <span class="text-[9px] text-sky-400 font-bold bg-sky-500/10 border border-sky-500/20 px-1 py-0.5 rounded">烟雾报警</span>
              {% else %}
              <span class="text-[9px] text-rose-400 font-bold bg-rose-500/10 border border-rose-500/20 px-1 py-0.5 rounded">火灾报警</span>
              {% endif %}
            </div>
          </div>
        {% else %}
          <div class="w-[calc(20%-10px)] min-w-[160px] shrink-0 rounded-xl bg-slate-950/40 border border-slate-800/60 p-2 flex flex-col gap-1.5 opacity-60">
            <div class="relative aspect-video rounded-lg overflow-hidden border border-slate-800/45 bg-slate-900/60 flex items-center justify-center">
              <svg class="w-8 h-8 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
              </svg>
            </div>
            <div class="flex justify-between items-center text-[10px] mt-0.5">
              <span class="text-slate-500 font-mono">--:--:--</span>
              <span class="text-slate-500 truncate max-w-[80px] font-medium">常规监控通道</span>
            </div>
            <div class="flex justify-between items-center mt-1">
              <span class="text-[9px] text-slate-500">通道-0{{ i + 1 }}</span>
              <span class="text-[9px] text-slate-500 font-semibold bg-slate-800/40 border border-slate-700/30 px-1 py-0.5 rounded">常规监控</span>
            </div>
          </div>
        {% endif %}
      {% endfor %}
    </div>
  </div>
</section>

<!-- Right Column: 核心数据指标 + 警情分析 + 趋势 -->
<section class="col-span-3 flex flex-col gap-4 h-full overflow-hidden">
  <!-- 核心数据指标 -->
  <div class="glass-panel rounded-2xl p-4 shadow-[0_8px_32px_rgba(0,0,0,0.37)] flex flex-col gap-3 shrink-0 border border-slate-800/80">
    <h2 class="text-sm font-bold tracking-wider text-slate-200 flex items-center gap-2">
      <span class="w-1.5 h-3.5 bg-cyan-400 rounded-full shadow-[0_0_8px_#22d3ee]"></span>
      核心数据指标
    </h2>
    <div class="grid grid-cols-2 gap-2.5 text-center">
      <!-- Row 1 -->
      <div onclick="filterByTimeRange('today')" class="cursor-pointer bg-slate-950/60 border border-slate-800/50 rounded-xl p-3 flex flex-col gap-1.5 relative overflow-hidden group hover:border-cyan-500/40 hover:shadow-[0_0_12px_rgba(6,182,212,0.3)] transition duration-300">
        <div class="flex justify-between items-center">
          <span class="text-slate-400 text-xs font-semibold tracking-wider">今日警情</span>
          <span class="text-rose-500 text-xs">🔔</span>
        </div>
        <span class="text-2xl font-black text-cyan-400 font-mono text-left" id="statToday">{{ today_count }}</span>
        <div class="absolute inset-x-0 bottom-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-40 group-hover:opacity-100 transition-opacity"></div>
      </div>
      <div onclick="filterByTimeRange('week')" class="cursor-pointer bg-slate-950/60 border border-slate-800/50 rounded-xl p-3 flex flex-col gap-1.5 relative overflow-hidden group hover:border-orange-500/40 hover:shadow-[0_0_12px_rgba(249,115,22,0.3)] transition duration-300">
        <div class="flex justify-between items-center">
          <span class="text-slate-400 text-xs font-semibold tracking-wider">本周警情</span>
          <span class="text-cyan-500 text-xs">📅</span>
        </div>
        <span class="text-2xl font-black text-orange-400 font-mono text-left" id="statWeek">{{ week_count }}</span>
        <div class="absolute inset-x-0 bottom-0 h-[2px] bg-gradient-to-r from-transparent via-orange-500 to-transparent opacity-40 group-hover:opacity-100 transition-opacity"></div>
      </div>
      <!-- Row 2 -->
      <div class="bg-slate-950/60 border border-slate-800/50 rounded-xl p-3 flex flex-col gap-1.5 relative overflow-hidden group hover:border-emerald-500/40 transition duration-300">
        <div class="flex justify-between items-center">
          <span class="text-slate-400 text-xs font-semibold tracking-wider">在线设备</span>
          <span class="text-emerald-500 text-xs">🖥️</span>
        </div>
        <span class="text-2xl font-black text-emerald-400 font-mono text-left" id="statOnline">{{ online_count|default(0) }}</span>
        <div class="absolute inset-x-0 bottom-0 h-[2px] bg-gradient-to-r from-transparent via-emerald-500 to-transparent opacity-40 group-hover:opacity-100 transition-opacity"></div>
      </div>
      <div class="bg-slate-950/60 border border-slate-800/50 rounded-xl p-3 flex flex-col gap-1.5 relative overflow-hidden group hover:border-slate-500/40 transition duration-300">
        <div class="flex justify-between items-center">
          <span class="text-slate-400 text-xs font-semibold tracking-wider">离线设备</span>
          <span class="text-slate-500 text-xs">📶</span>
        </div>
        <span class="text-2xl font-black text-slate-400 font-mono text-left" id="statOffline">{{ offline_count|default(0) }}</span>
        <div class="absolute inset-x-0 bottom-0 h-[2px] bg-gradient-to-r from-transparent via-slate-500 to-transparent opacity-40 group-hover:opacity-100 transition-opacity"></div>
      </div>
      <!-- Row 3 -->
      <div class="bg-slate-950/60 border border-slate-800/50 rounded-xl p-3 flex flex-col gap-1.5 relative overflow-hidden group hover:border-blue-500/40 hover:shadow-[0_0_12px_rgba(59,130,246,0.3)] transition duration-300">
        <div class="flex justify-between items-center">
          <span class="text-slate-400 text-xs font-semibold tracking-wider">AI处理率</span>
          <span class="text-purple-500 text-xs">⚙️</span>
        </div>
        <span class="text-2xl font-black text-blue-400 font-mono text-left" id="statAiRate">{{ ai_rate|default(100) }}<span class="text-sm">%</span></span>
        <div class="absolute inset-x-0 bottom-0 h-[2px] bg-gradient-to-r from-transparent via-blue-500 to-transparent opacity-40 group-hover:opacity-100 transition-opacity"></div>
      </div>
      <div onclick="filterByTimeRange('pending')" class="cursor-pointer bg-slate-950/60 border border-slate-800/50 rounded-xl p-3 flex flex-col gap-1.5 relative overflow-hidden group hover:border-rose-500/40 hover:shadow-[0_0_12px_rgba(244,63,94,0.3)] transition duration-300">
        <div class="flex justify-between items-center">
          <span class="text-slate-400 text-xs font-semibold tracking-wider">待处理</span>
          <span class="text-rose-500 text-xs">⚠️</span>
        </div>
        <span class="text-2xl font-black text-rose-400 font-mono text-left animate-pulse" id="statPending">{{ pending_alarms }}</span>
        <div class="absolute inset-x-0 bottom-0 h-[2px] bg-gradient-to-r from-transparent via-rose-500 to-transparent opacity-40 group-hover:opacity-100 transition-opacity"></div>
      </div>
    </div>
  </div>

  <!-- 地区报警排行 -->
  <div class="glass-panel rounded-2xl p-4 shadow-[0_8px_32px_rgba(0,0,0,0.37)] flex flex-col gap-2 shrink-0 border border-slate-800/80">
    <h2 class="text-sm font-bold tracking-wider text-slate-200 flex items-center gap-2">
      <span class="w-1.5 h-3.5 bg-amber-500 rounded-full shadow-[0_0_8px_#f59e0b]"></span>
      地区报警排行 (TOP 5)
    </h2>
    <div class="flex flex-col gap-2.5 mt-1" id="rankingContainer">
      {% for r in monthly_ranking %}
      <div class="flex flex-col gap-1 text-xs">
        <div class="flex justify-between items-center text-slate-350">
          <span class="font-medium text-slate-200">{{ r.name }}</span>
          <span class="font-mono text-cyan-400 font-bold">{{ r.count }} 次</span>
        </div>
        <div class="w-full bg-slate-950/60 rounded-full h-1.5 overflow-hidden border border-slate-900">
          <div class="bg-gradient-to-r from-cyan-500 to-blue-500 h-full rounded-full" style="width: {{ (r.count / max_rank * 100)|round|int }}%"></div>
        </div>
      </div>
      {% else %}
      <div class="text-center text-slate-500 py-4">暂无排行数据</div>
      {% endfor %}
    </div>
  </div>

  <!-- 警情类型分析 -->
  <div class="glass-panel rounded-2xl p-4 shadow-[0_8px_32px_rgba(0,0,0,0.37)] flex flex-col gap-2 shrink-0 border border-slate-800/80">
    <h2 class="text-sm font-bold tracking-wider text-slate-200 flex items-center gap-2">
      <span class="w-1.5 h-3.5 bg-purple-500 rounded-full shadow-[0_0_8px_#a855f7]"></span>
      警情类型分析
    </h2>
    <div class="flex gap-3 h-[130px]">
      <div class="w-[120px] shrink-0 relative" id="pieChartContainer">
        <div id="accuracyChart" class="w-full h-full"></div>
        <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div class="text-center">
            <span class="text-2xl font-black text-slate-100 block" id="pieCenterTotal">{{ total_alarms }}</span>
            <span class="text-[10px] text-slate-500">总计</span>
          </div>
        </div>
      </div>
      {% set total_stats = true_count + false_count + missed_count + pending_alarms %}
      {% set true_pct = (true_count / total_stats * 100)|round|int if total_stats > 0 else 0 %}
      {% set false_pct = (false_count / total_stats * 100)|round|int if total_stats > 0 else 0 %}
      {% set missed_pct = (missed_count / total_stats * 100)|round|int if total_stats > 0 else 0 %}
      {% set pending_pct = (pending_alarms / total_stats * 100)|round|int if total_stats > 0 else 0 %}
      <div class="flex-1 flex flex-col justify-center gap-1.5 text-[11px] min-w-0 pr-1">
        <div class="flex items-center justify-between whitespace-nowrap">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-sm bg-rose-500 shadow-[0_0_4px_#f43f5e]"></span>
            <span class="text-slate-400">确认火警</span>
          </div>
          <div class="flex items-center gap-0.5 font-mono">
            <span class="text-rose-400 font-bold" id="pctTrue">{{ true_pct }}%</span>
            <span class="text-slate-500 text-[10px]" id="statTrueCount">({{ true_count }}次)</span>
          </div>
        </div>
        <div class="flex items-center justify-between whitespace-nowrap">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-sm bg-emerald-500 shadow-[0_0_4px_#10b981]"></span>
            <span class="text-slate-400">误报记录</span>
          </div>
          <div class="flex items-center gap-0.5 font-mono">
            <span class="text-emerald-400 font-bold" id="pctFalse">{{ false_pct }}%</span>
            <span class="text-slate-500 text-[10px]" id="statFalseCount">({{ false_count }}次)</span>
          </div>
        </div>
        <div class="flex items-center justify-between whitespace-nowrap">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-sm bg-amber-500 shadow-[0_0_4px_#f59e0b]"></span>
            <span class="text-slate-400">漏报记录</span>
          </div>
          <div class="flex items-center gap-0.5 font-mono">
            <span class="text-amber-400 font-bold" id="pctMissed">{{ missed_pct }}%</span>
            <span class="text-slate-500 text-[10px]" id="statMissedCount">({{ missed_count }}次)</span>
          </div>
        </div>
        <div class="flex items-center justify-between whitespace-nowrap">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-sm bg-cyan-500 shadow-[0_0_4px_#06b6d4]"></span>
            <span class="text-slate-400">待处理</span>
          </div>
          <div class="flex items-center gap-0.5 font-mono">
            <span class="text-cyan-400 font-bold" id="pctPending">{{ pending_pct }}%</span>
            <span class="text-slate-500 text-[10px]" id="statPendingCount">({{ pending_alarms }}次)</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- 30天警情趋势 -->
  <div class="flex-1 glass-panel rounded-2xl p-4 shadow-[0_8px_32px_rgba(0,0,0,0.37)] flex flex-col gap-2 overflow-hidden border border-slate-800/80 min-h-0">
    <h2 class="text-sm font-bold tracking-wider text-slate-200 flex items-center gap-2 shrink-0">
      <span class="w-1.5 h-3.5 bg-cyan-400 rounded-full shadow-[0_0_8px_#22d3ee]"></span>
      30天警情趋势
    </h2>
    <div id="trendChart" class="flex-1 w-full min-h-0"></div>
  </div>
</section>
</main>

<script>
// Digital Clock & Date Update
function upd(){
  var n=new Date();
  var clockEl = document.getElementById('clock');
  var dateEl = document.getElementById('liveDate');
  if (clockEl) clockEl.textContent=n.toLocaleTimeString('zh-CN',{hour12:false});
  if (dateEl) dateEl.textContent=n.getFullYear()+'年'+(n.getMonth()+1).toString().padStart(2,'0')+'月'+n.getDate().toString().padStart(2,'0')+'日 星期'+['日','一','二','三','四','五','六'][n.getDay()];
}
upd();
setInterval(upd,1000);

// WebSocket connection and Dynamic Discovery Port Scanner
var ws=null,wsReconnectTimer=null,wsReconnectDelay=1000;
var discoveredCameras = {}; // camera_id -> {id, name, port, location, host}
var scanningIntervalId = null;

// Do not pre-populate from database to ensure a clean slate 0-device start.
// Devices will only appear once successfully discovered and connected via the port scanner.

// Initialize Scanner on load
function initScanner() {
  updateCameraSelectorUI();
  
  const scanIntervalSec = parseInt(document.getElementById('scanIntervalInput').value) || 1;
  if (scanningIntervalId) clearInterval(scanningIntervalId);
  
  // Perform initial scan
  scanActivePorts();
  
  // Schedule scanning
  scanningIntervalId = setInterval(scanActivePorts, scanIntervalSec * 1000);
}

function triggerDefaultScan() {
  console.log('[Scanner] Starting default scan on 0.0.0.0...');
  scanActivePorts('0.0.0.0');
}

function triggerManualScan() {
  const host = document.getElementById('scanHost').value.trim() || '0.0.0.0';
  console.log('[Scanner] Starting manual scan on IP:', host);
  scanActivePorts(host);
}

function scanActivePorts(targetHost) {
  const defaultHost = document.getElementById('scanHost').value.trim() || '0.0.0.0';
  const hostToScan = targetHost || defaultHost;
  
  const start = parseInt(document.getElementById('scanStart').value) || 9991;
  const end = parseInt(document.getElementById('scanEnd').value) || 9994;
  
  for (let port = start; port <= end; port++) {
    if (isPortBeingUsed(hostToScan, port)) continue;
    checkPortWS(hostToScan, port);
  }
}

function isPortBeingUsed(host, port) {
  const selector = document.getElementById('cameraSelector');
  if (selector && selector.selectedIndex >= 0 && selector.options[selector.selectedIndex]) {
    const activePort = parseInt(selector.options[selector.selectedIndex].getAttribute('data-port'));
    const activeHost = selector.options[selector.selectedIndex].getAttribute('data-host') || '127.0.0.1';
    if (activePort === port && activeHost === host && ws && ws.readyState === WebSocket.OPEN) {
      return true;
    }
  }
  return false;
}

function checkPortWS(host, port) {
  const testUrl = 'ws://' + host + ':' + port;
  const socket = new WebSocket(testUrl);
  
  const timeoutId = setTimeout(() => {
    if (socket.readyState !== WebSocket.OPEN) {
      socket.close();
    }
  }, 1200);

  socket.onopen = function() {
    clearTimeout(timeoutId);
  };

  socket.onmessage = function(event) {
    clearTimeout(timeoutId);
    if (typeof event.data === 'string') {
      try {
        const info = JSON.parse(event.data);
        if (info.type === 'camera_info') {
          console.log('[Scanner] Discovered camera:', info);
          registerDiscoveredCamera(info, host);
          socket.close();
        }
      } catch (err) {
        socket.close();
      }
    } else {
      socket.close();
    }
  };

  socket.onerror = function() {
    clearTimeout(timeoutId);
  };

  socket.onclose = function() {
    clearTimeout(timeoutId);
  };
}

function registerDiscoveredCamera(info, host) {
  const camId = info.camera_id;
  const camName = info.camera_name;
  const port = info.ws_port;
  const loc = info.location;

  const existing = discoveredCameras[camId];
  if (existing && existing.port === port && existing.host === host && existing.name === camName && existing.location === loc) {
    if (scanningIntervalId) {
      console.log('[Scanner] Camera already exists. Stopping auto-scanning interval.');
      clearInterval(scanningIntervalId);
      scanningIntervalId = null;
    }
    return;
  }

  discoveredCameras[camId] = {
    id: camId,
    name: camName,
    port: port,
    location: loc,
    host: host
  };

  fetch('/api/camera/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      camera_id: camId,
      ws_port: port,
      ip: host,
      location: loc,
      camera_name: camName
    })
  })
  .then(res => res.json())
  .then(data => {
    console.log('[Scanner] Camera registered on server:', data);
    updateCameraSelectorUI(camId);
    
    if (scanningIntervalId) {
      console.log('[Scanner] Camera successfully registered. Stopping auto-scanning interval.');
      clearInterval(scanningIntervalId);
      scanningIntervalId = null;
    }
  })
  .catch(err => console.error('[Scanner] Failed to register camera:', err));
}

function updateCameraSelectorUI(selectedId) {
  const selector = document.getElementById('cameraSelector');
  if (!selector) return;

  const currentSelection = selector.value;
  selector.innerHTML = '';

  const cameraKeys = Object.keys(discoveredCameras);
  if (cameraKeys.length === 0) {
    const opt = document.createElement('option');
    opt.value = "";
    opt.textContent = "🔍 正在检索监控...";
    opt.disabled = true;
    opt.selected = true;
    selector.appendChild(opt);
    return;
  }

  cameraKeys.forEach(id => {
    const cam = discoveredCameras[id];
    const opt = document.createElement('option');
    opt.value = cam.id;
    opt.setAttribute('data-port', cam.port);
    opt.setAttribute('data-host', cam.host || '127.0.0.1');
    if (cam.name === cam.location || !cam.location) {
      opt.textContent = `${cam.name} (Port ${cam.port})`;
    } else {
      opt.textContent = `${cam.name} (${cam.location} - Port ${cam.port})`;
    }
    selector.appendChild(opt);
  });

  if (discoveredCameras[currentSelection]) {
    selector.value = currentSelection;
  } else if (selectedId && discoveredCameras[selectedId]) {
    selector.value = selectedId;
    changeCameraStream();
  } else {
    selector.selectedIndex = 0;
    changeCameraStream();
  }
}

function changeCameraStream() {
  wsConnect();
}

function wsConnect() {
  if(wsReconnectTimer){clearTimeout(wsReconnectTimer); wsReconnectTimer=null;}
  if(ws){try{ws.close();}catch(e){}}

  const selector = document.getElementById('cameraSelector');
  if(!selector || selector.selectedIndex < 0 || !selector.value) {
    document.getElementById('videoOffline').style.display='flex';
    document.getElementById('cameraFrame').classList.add('hidden');
    var t=document.getElementById('videoTag');
    if (t) {
      t.classList.remove('hidden');
      t.textContent='No Signal';
      t.className='absolute top-3 left-3 bg-slate-500/10 border border-slate-500/20 text-slate-400 text-[9px] font-bold px-2.5 py-0.5 rounded uppercase tracking-wider shadow-md';
    }
    return;
  }

  const host = selector.options[selector.selectedIndex].getAttribute('data-host') || location.hostname || '127.0.0.1';
  const port = selector.options[selector.selectedIndex].getAttribute('data-port') || '9999';
  const wsUrl = 'ws://' + host + ':' + port;

  console.log('[WS] Connecting stream to: ' + wsUrl);
  ws=new WebSocket(wsUrl);
  ws.binaryType='blob';

  ws.onopen=function(){
    console.log('[WS] Stream connected: ' + wsUrl);
    document.getElementById('videoOffline').style.display='none';
    document.getElementById('cameraFrame').classList.remove('hidden');
    var t=document.getElementById('videoTag');
    if (t) {
      t.classList.remove('hidden');
      t.textContent='Live';
      t.className='absolute top-3 left-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[9px] font-bold px-2.5 py-0.5 rounded uppercase tracking-wider shadow-md';
    }
    wsReconnectDelay=1000;
    
    if (scanningIntervalId) {
      console.log('[WS] Stream connected. Stopping auto-scanning interval.');
      clearInterval(scanningIntervalId);
      scanningIntervalId = null;
    }
  };

  ws.onmessage=function(e){
    if (typeof e.data === 'string') {
      try {
        const info = JSON.parse(e.data);
        if (info.type === 'camera_info') {
          console.log('[WS] Stream info handshake:', info);
          registerDiscoveredCamera(info);
        }
      } catch(err) {
        console.error('[WS] Parse message error:', err);
      }
      return;
    }

    var u=URL.createObjectURL(e.data);
    var img=document.getElementById('cameraFrame');
    img.onload=function(){URL.revokeObjectURL(u);};
    img.src=u;
  };

  ws.onclose=function(e){
    console.log('[WS] Stream closed. Code:', e.code);
    document.getElementById('videoOffline').style.display='flex';
    document.getElementById('cameraFrame').classList.add('hidden');
    var t=document.getElementById('videoTag');
    if (t) {
      t.classList.remove('hidden');
      t.textContent='Offline';
      t.className='absolute top-3 left-3 bg-rose-500/10 border border-rose-500/20 text-rose-455 text-[9px] font-bold px-2.5 py-0.5 rounded uppercase tracking-wider shadow-md';
    }

    wsReconnectTimer=setTimeout(wsConnect, wsReconnectDelay);
    wsReconnectDelay=Math.min(wsReconnectDelay*2, 10000);
  };

  ws.onerror=function(e){
    console.error('[WS] Stream error observed:', e);
  };
}

// Center View switching (Video / Map)
var map = null;
var cameraMarkers = [];
var alarmMarkers = [];

function switchCenterView(type) {
  const videoFeed = document.getElementById('videoFeedContainer');
  const mapContainer = document.getElementById('mapContainer');
  const btnVideo = document.getElementById('btnViewVideo');
  const btnMap = document.getElementById('btnViewMap');
  
  if (type === 'video') {
    if (videoFeed) videoFeed.classList.remove('hidden');
    if (mapContainer) mapContainer.classList.add('hidden');
    if (btnVideo) btnVideo.className = 'px-2.5 py-1 rounded-md text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-sm transition active:scale-95';
    if (btnMap) btnMap.className = 'px-2.5 py-1 rounded-md text-[10px] font-semibold text-slate-400 hover:text-slate-200 transition active:scale-95';
  } else {
    if (videoFeed) videoFeed.classList.add('hidden');
    if (mapContainer) mapContainer.classList.remove('hidden');
    if (btnVideo) btnVideo.className = 'px-2.5 py-1 rounded-md text-[10px] font-semibold text-slate-400 hover:text-slate-200 transition active:scale-95';
    if (btnMap) btnMap.className = 'px-2.5 py-1 rounded-md text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-sm transition active:scale-95';
    
    initMapOnce();
  }
}

function initMapOnce() {
  if (map) {
    setTimeout(() => { map.invalidateSize(); }, 200);
    return;
  }
  
  // Chongqing coordinate center
  const centerLat = 29.452537;
  const centerLng = 106.529813;
  
  map = L.map('mapContainer', {
    zoomControl: false
  }).setView([centerLat, centerLng], 16);
  
  L.control.zoom({ position: 'bottomright' }).addTo(map);
  
  // Use CartoDB Dark Matter tiles (looks extremely tech-y and dark)
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO'
  }).addTo(map);
  
  updateMapMarkers();
  setTimeout(() => { map.invalidateSize(); }, 200);
}

function updateMapMarkers() {
  if (!map) return;
  
  // Clear old markers
  cameraMarkers.forEach(m => map.removeLayer(m));
  alarmMarkers.forEach(m => map.removeLayer(m));
  cameraMarkers = [];
  alarmMarkers = [];
  
  // Custom Icon Helpers
  const cameraIcon = L.divIcon({
    html: `<div class="w-8 h-8 rounded-full bg-cyan-500/20 border-2 border-cyan-400 flex items-center justify-center text-sm shadow-[0_0_8px_#22d3ee]">📷</div>`,
    className: 'custom-map-icon',
    iconSize: [32, 32],
    iconAnchor: [16, 16]
  });
  
  const fireIcon = L.divIcon({
    html: `<div class="w-10 h-10 rounded-full bg-rose-500/30 border border-rose-500 flex items-center justify-center text-base shadow-[0_0_12px_#f43f5e] animate-pulse">🔥</div>`,
    className: 'custom-map-icon',
    iconSize: [40, 40],
    iconAnchor: [20, 20]
  });

  const smokeIcon = L.divIcon({
    html: `<div class="w-10 h-10 rounded-full bg-sky-500/30 border border-sky-500 flex items-center justify-center text-base shadow-[0_0_12px_#0ea5e9] animate-pulse">🌫️</div>`,
    className: 'custom-map-icon',
    iconSize: [40, 40],
    iconAnchor: [20, 20]
  });
  
  // Render Cameras
  globalCamerasList.forEach(c => {
    // If camera doesn't have coordinates, default to center campus coordinates
    const lat = parseFloat(c.Latitude) || 29.452537;
    const lng = parseFloat(c.Longitude) || 106.529813;
    
    const popupHtml = `
      <div class="p-2 text-xs flex flex-col gap-1.5 min-w-[200px]">
        <h4 class="text-sm font-bold text-cyan-400 flex items-center gap-1">📷 ${c.Name}</h4>
        <div class="flex flex-col gap-0.5 text-slate-350">
          <div><span class="text-slate-500">IP:</span> ${c.IP || '--'}</div>
          <div><span class="text-slate-500">MAC:</span> ${c.MAC || '--'}</div>
          <div><span class="text-slate-500">型号:</span> ${c.Type || '--'}</div>
          <div><span class="text-slate-500">区域:</span> ${c.AreaName || '重庆市'}</div>
        </div>
      </div>
    `;
    
    const marker = L.marker([lat, lng], { icon: cameraIcon }).bindPopup(popupHtml).addTo(map);
    cameraMarkers.push(marker);
  });
  
  // Render Recent Alarms
  globalRecentAlarmsList.forEach((a, idx) => {
    const cam = globalCamerasList.find(c => c.Id === a.CameraId);
    let lat = parseFloat(a.Latitude) || (cam ? parseFloat(cam.Latitude) : null) || 29.452537;
    let lng = parseFloat(a.Longitude) || (cam ? parseFloat(cam.Longitude) : null) || 106.529813;
    
    // Add a tiny spiral/circular offset to prevent total overlap of markers at the same location
    const angle = idx * 0.5;
    const radius = 0.00004 + idx * 0.00001;
    lat += Math.sin(angle) * radius;
    lng += Math.cos(angle) * radius;
    
    const isSmoke = a.Description && a.Description.includes('烟雾') && !a.Description.includes('火焰');
    const typeLabel = isSmoke ? '疑似烟雾预警' : '火焰预警记录';
    const headerClass = isSmoke ? 'text-sky-400' : 'text-rose-450';
    const pulseBg = isSmoke ? 'bg-sky-500 shadow-[0_0_6px_#0ea5e9]' : 'bg-rose-500 shadow-[0_0_6px_#f43f5e]';
    const btnClass = isSmoke ? 'bg-sky-600 hover:bg-sky-500' : 'bg-rose-600 hover:bg-rose-500';
    const activeIcon = isSmoke ? smokeIcon : fireIcon;
    
    let mediaHtml = '';
    if (a.Picture) {
      mediaHtml += `<img src="${a.Picture}" class="w-full aspect-video object-cover rounded-lg border border-slate-800/80 my-1.5 shadow-md hover:scale-105 transition duration-300">`;
    }
    if (a.VideoUrl) {
      mediaHtml += `
        <video src="${a.VideoUrl}" class="w-full aspect-video object-cover rounded-lg border border-slate-800/80 my-1.5 shadow-md" controls autoplay muted loop></video>
      `;
    }
    
    const popupHtml = `
      <div class="p-2 text-xs flex flex-col gap-1.5 min-w-[220px]">
        <h4 class="text-sm font-bold ${headerClass} flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full ${pulseBg} animate-pulse"></span>
          ${isSmoke ? '🌫️' : '🔥'} ${typeLabel}
        </h4>
        <div class="flex flex-col gap-0.5 text-slate-350">
          <div><span class="text-slate-500">位置:</span> ${a.Location || a.AreaName || '未知位置'}</div>
          <div><span class="text-slate-500">时间:</span> <span class="font-mono">${a.CreatTime || '--'}</span></div>
          <div><span class="text-slate-500">置信度:</span> <span class="${isSmoke ? 'text-sky-400' : 'text-rose-400'} font-bold">${a.Confidence ? Math.round(a.Confidence * 100) + '%' : '95%'}</span></div>
        </div>
        ${mediaHtml}
        <button onclick="showAlarmDetail(${a.Id})" class="w-full ${btnClass} text-white py-1.5 rounded-lg font-bold transition active:scale-95 text-center mt-1">
          立即处理
        </button>
      </div>
    `;
    
    const marker = L.marker([lat, lng], { icon: activeIcon }).bindPopup(popupHtml).addTo(map);
    alarmMarkers.push(marker);
  });
}

// Search / Query & Interactive Filtering functionality
var filterLocation = '';
var filterStart = '';
var filterEnd = '';
var filterStatus = ''; // '' for all, '1' for pending

// Global lists populated on load and kept in sync
var globalCamerasList = [
  {% for c in cameras %}
  {
    Id: {{ c.Id }},
    Name: "{{ c.Name|replace('"', '\\"') }}",
    IP: "{{ c.IP or '' }}",
    MAC: "{{ c.MAC or '' }}",
    Type: "{{ c.Type or '' }}",
    AreaName: "{{ c.AreaName or '' }}",
    Latitude: "{{ c.Latitude or '' }}",
    Longitude: "{{ c.Longitude or '' }}",
    status: "{{ c.status or 'offline' }}"
  },
  {% endfor %}
];

var globalRecentAlarmsList = [
  {% for a in recent_alarms %}
  {
    Id: {{ a.Id }},
    CreatTime: "{{ a.CreatTime or '' }}",
    Location: "{{ (a.Location or a.CameraName or '未知位置')|replace('\\\\', '\\\\\\\\')|replace('"', '\\"')|replace('\r', '')|replace('\n', '\\n') }}",
    Picture: "{{ a.Picture or '' }}",
    VideoUrl: "{{ a.VideoUrl or '' }}",
    Confidence: {{ a.Confidence or 0.95 }},
    CameraId: {{ a.CameraId or 1 }},
    CameraName: "{{ a.CameraName or '' }}",
    AreaName: "{{ a.AreaName or '' }}"
  },
  {% endfor %}
];
var filterStart = '';
var filterEnd = '';
var filterStatus = ''; // '' for all, '1' for pending

var earliestAlarmTime = "{{ earliest_time }}";
function formatDateTimeLocal(dateStr) {
  if (!dateStr) return "";
  return dateStr.replace(' ', 'T').substring(0, 16);
}
function getNowDateTimeLocal() {
  const now = new Date();
  const tzOffset = now.getTimezoneOffset() * 60000;
  return (new Date(now - tzOffset)).toISOString().slice(0, 16);
}

// Bind event listeners for inline filtering on load
document.addEventListener("DOMContentLoaded", function() {
  const startInput = document.getElementById('startTime');
  const endInput = document.getElementById('endTime');
  const locInput = document.getElementById('searchLocation');
  
  if (startInput) {
    startInput.value = '';
    filterStart = '';
    startInput.addEventListener('change', function() {
      filterStart = this.value;
      fetchRealtimeData();
    });
  }
  if (endInput) {
    endInput.value = '';
    filterEnd = '';
    endInput.addEventListener('change', function() {
      filterEnd = this.value;
      fetchRealtimeData();
    });
  }
  if (locInput) {
    locInput.addEventListener('input', function() {
      filterLocation = this.value.trim().toLowerCase();
      fetchRealtimeData();
    });
  }
  
  const btnAll = document.getElementById('filterBtnAll');
  if (btnAll) {
    btnAll.classList.remove('border-slate-800/80', 'text-slate-400');
    btnAll.classList.add('border-white', 'text-white', 'bg-slate-800/50');
  }
});

function filterByTimeRange(range) {
  const startInput = document.getElementById('startTime');
  const endInput = document.getElementById('endTime');
  const locInput = document.getElementById('searchLocation');
  
  filterStatus = ''; // reset status by default
  const now = new Date();
  
  if (range === 'today') {
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    if (startInput) startInput.value = formatDateTimeLocal(todayStart.toISOString());
    if (endInput) endInput.value = getNowDateTimeLocal();
  } else if (range === 'week') {
    const weekStart = new Date(now.getTime() - 7 * 24 * 3600 * 1000);
    if (startInput) startInput.value = formatDateTimeLocal(weekStart.toISOString());
    if (endInput) endInput.value = getNowDateTimeLocal();
  } else if (range === 'month') {
    const monthStart = new Date(now.getTime() - 30 * 24 * 3600 * 1000);
    if (startInput) startInput.value = formatDateTimeLocal(monthStart.toISOString());
    if (endInput) endInput.value = getNowDateTimeLocal();
  } else if (range === 'year') {
    const yearStart = new Date(now.getFullYear(), 0, 1);
    if (startInput) startInput.value = formatDateTimeLocal(yearStart.toISOString());
    if (endInput) endInput.value = getNowDateTimeLocal();
  } else if (range === 'pending') {
    filterStatus = 'pending';
    if (startInput) startInput.value = '';
    if (endInput) endInput.value = '';
  } else if (range === 'processed') {
    filterStatus = 'processed';
    if (startInput) startInput.value = '';
    if (endInput) endInput.value = '';
  } else if (range === 'false_alarm') {
    filterStatus = 'false_alarm';
    if (startInput) startInput.value = '';
    if (endInput) endInput.value = '';
  } else if (range === 'missed') {
    filterStatus = 'missed';
    if (startInput) startInput.value = '';
    if (endInput) endInput.value = '';
  } else if (range === 'all') {
    filterStatus = '';
    if (startInput) startInput.value = '';
    if (endInput) endInput.value = '';
    if (locInput) locInput.value = '';
    filterLocation = '';
  }
  
  // Update active button state styling
  const btns = {
    'all': 'filterBtnAll',
    'today': 'filterBtnToday',
    'week': 'filterBtnWeek',
    'pending': 'filterBtnPending',
    'processed': 'filterBtnProcessed',
    'false_alarm': 'filterBtnFalse',
    'missed': 'filterBtnMissed'
  };
  Object.keys(btns).forEach(key => {
    const btn = document.getElementById(btns[key]);
    if (btn) {
      if (key === range) {
        btn.classList.remove('border-slate-800/80', 'text-slate-400');
        btn.classList.add('border-white', 'text-white', 'bg-slate-800/50');
      } else {
        btn.classList.remove('border-white', 'text-white', 'bg-slate-800/50');
        btn.classList.add('border-slate-800/80', 'text-slate-400');
      }
    }
  });

  filterStart = startInput ? startInput.value : '';
  filterEnd = endInput ? endInput.value : '';
  fetchRealtimeData();
}

function filterByLocation(loc) {
  const locInput = document.getElementById('searchLocation');
  if (locInput) {
    locInput.value = loc;
    filterLocation = loc.toLowerCase();
    fetchRealtimeData();
  }
}

function clearAllAlarms() {
  if (!confirm('是否将所有未处理报警标记为“误报无需处理”？')) return;
  fetch('/admin/alarm/clear_all', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
      if (data.code === 200) {
        alert('所有未处理报警已一键处理为误报！');
        fetchRealtimeData();
      } else {
        alert('处理失败: ' + data.msg);
      }
    })
    .catch(err => {
      console.error(err);
      alert('网络请求失败');
    });
}

var trendChartInstance = null;
var accuracyChartInstance = null;
var trendOption = null;
var accuracyOption = null;

// Global alarmData mapping for details modal
const alarmData = {
  {% for a in recent_alarms %}
  "{{ a.Id }}": {
    id: {{ a.Id }},
    time: "{{ a.CreatTime or '' }}",
    location: "{{ (a.Location or a.CameraName or '未知位置')|replace('\\\\', '\\\\\\\\')|replace('"', '\\"')|replace('\r', '')|replace('\n', '\\n') }}",
    picture: "{{ a.Picture or '' }}",
    videoUrl: "{{ a.VideoUrl or '' }}",
    status: "{{ a.Status }}",
    desc: "{{ (a.Description or '')|replace('\\\\', '\\\\\\\\')|replace('"', '\\"')|replace('\r', '')|replace('\n', '\\n') }}",
    urgency: "{{ a.UrgencyDegree or '' }}",
    result: "{{ a.OperateResult or '' }}",
    operator: "{{ a.OperatorName or '' }}",
    operateTime: "{{ a.OperateTime or '' }}"
  },
  {% endfor %}
};

function showAlarmDetail(id) {
  const a = alarmData[id];
  if (!a) return;

  const form = document.getElementById('modalProcessForm');
  if (form) form.action = '/admin/alarm/process/' + id;
  
  const timeEl = document.getElementById('modalAlarmTime');
  if (timeEl) timeEl.textContent = a.time || '--';
  
  const locEl = document.getElementById('modalAlarmLocation');
  if (locEl) locEl.textContent = a.location || '未知位置';
  
  const typeEl = document.getElementById('modalAlarmType');
  if (typeEl) {
    const isSmoke = a.desc && a.desc.includes('烟雾') && !a.desc.includes('火焰');
    if (isSmoke) {
      typeEl.innerHTML = '<span class="text-sky-400 font-bold bg-sky-500/10 border border-sky-500/20 px-1.5 py-0.5 rounded">烟雾报警 (疑似火灾)</span>';
    } else {
      typeEl.innerHTML = '<span class="text-rose-400 font-bold bg-rose-500/10 border border-rose-500/20 px-1.5 py-0.5 rounded">火灾报警</span>';
    }
  }
  
  // Picture
  const img = document.getElementById('modalAlarmImage');
  const noImg = document.getElementById('modalAlarmNoImage');
  if (img && noImg) {
    if (a.picture) {
      img.src = a.picture;
      img.classList.remove('hidden');
      noImg.classList.add('hidden');
    } else {
      img.src = '';
      img.classList.add('hidden');
      noImg.classList.remove('hidden');
    }
  }
  
  // Video
  const video = document.getElementById('modalAlarmVideo');
  const noVideo = document.getElementById('modalAlarmNoVideo');
  if (video && noVideo) {
    if (a.videoUrl) {
      video.src = a.videoUrl;
      video.load();
      video.play().catch(e => console.log("Video play failed:", e));
      video.classList.remove('hidden');
      noVideo.classList.add('hidden');
    } else {
      video.src = '';
      video.classList.add('hidden');
      noVideo.classList.remove('hidden');
    }
  }

  const statusEl = document.getElementById('modalAlarmStatus');
  if (statusEl) {
    if (a.status === '1') {
      statusEl.textContent = '待处理';
      statusEl.className = 'text-rose-400 font-bold';
      const pForm = document.getElementById('modalProcessForm');
      if (pForm) pForm.classList.remove('hidden');
      const pInfo = document.getElementById('modalProcessedInfo');
      if (pInfo) pInfo.classList.add('hidden');
    } else {
      let statusText = '已处理';
      let statusClass = 'text-emerald-400 font-bold';
      if (a.result === '误报无需处理') {
        statusText = '排除误报';
        statusClass = 'text-emerald-400 font-bold';
      } else if (a.result === '漏报记录') {
        statusText = '漏报记录';
        statusClass = 'text-amber-400 font-bold animate-pulse';
      } else {
        statusText = '已处理';
        statusClass = 'text-blue-400 font-bold';
      }
      statusEl.textContent = statusText;
      statusEl.className = statusClass;
      
      const pForm = document.getElementById('modalProcessForm');
      if (pForm) pForm.classList.add('hidden');
      const pInfo = document.getElementById('modalProcessedInfo');
      if (pInfo) pInfo.classList.remove('hidden');
      
      const opEl = document.getElementById('modalInfoOperator');
      if (opEl) opEl.textContent = a.operator || '系统/未知';
      const opTimeEl = document.getElementById('modalInfoTime');
      if (opTimeEl) opTimeEl.textContent = a.operateTime || '--';
      const opResEl = document.getElementById('modalInfoResult');
      if (opResEl) opResEl.textContent = a.result || '--';
      const opUrgEl = document.getElementById('modalInfoUrgency');
      if (opUrgEl) opUrgEl.textContent = a.urgency || '普通';
      const opDescEl = document.getElementById('modalInfoDesc');
      if (opDescEl) opDescEl.textContent = a.desc || '无备注';
      
      const auditActions = document.getElementById('modalAuditActions');
      if (auditActions) {
        if (a.status === '2') {
          auditActions.classList.remove('hidden');
          const appBtn = document.getElementById('modalAuditApproveBtn');
          if (appBtn) appBtn.href = '/admin/audit/approve/' + id;
          const rejBtn = document.getElementById('modalAuditRejectBtn');
          if (rejBtn) rejBtn.href = '/admin/audit/reject/' + id;
        } else {
          auditActions.classList.add('hidden');
        }
      }
    }
  }

  const modal = document.getElementById('detailModal');
  if (modal) modal.classList.remove('hidden');
}

function closeAlarmDetail() {
  const modal = document.getElementById('detailModal');
  if (modal) modal.classList.add('hidden');
  const video = document.getElementById('modalAlarmVideo');
  if (video) {
    video.pause();
  }
}

// 2s Polling for Real-Time Console Data
function fetchRealtimeData() {
  fetch('/api/stats')
    .then(response => {
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json();
    })
    .then(data => {
      // Update stats count indicators
      const stToday = document.getElementById('statToday');
      if (stToday) stToday.textContent = data.today_count;
      const stWeek = document.getElementById('statWeek');
      if (stWeek) stWeek.textContent = data.week_count;
      const stMonth = document.getElementById('statMonth');
      if (stMonth) stMonth.textContent = data.month_count;
      const stYear = document.getElementById('statYear');
      if (stYear) stYear.textContent = data.year_count;
      
      const statTotal = document.getElementById('statTotal');
      if (statTotal) statTotal.textContent = data.total;
      
      const statPending = document.getElementById('statPending');
      if (statPending) statPending.textContent = data.pending_alarms;
      
      const pendingBadge = document.getElementById('pendingAlarmsCount');
      if (pendingBadge) pendingBadge.textContent = data.pending_alarms;
      
      // Update global alarmData mapping dynamically
      (data.recent_alarms || []).forEach(a => {
        alarmData[a.Id] = {
          id: a.Id,
          time: a.CreatTime || '',
          location: a.Location || a.AreaName || '未知位置',
          picture: a.Picture || '',
          videoUrl: a.VideoUrl || '',
          status: a.Status,
          desc: a.Description || '',
          urgency: a.UrgencyDegree || '',
          result: a.OperateResult || '',
          operator: a.OperatorName || '',
          operateTime: a.OperateTime || ''
        };
      });

      // Update global map sync collections
      if (data.cameras) {
        globalCamerasList = data.cameras.map(c => ({
          Id: c.Id,
          Name: c.Name || '',
          IP: c.IP || '',
          MAC: c.MAC || '',
          Type: c.Type || '',
          AreaName: c.AreaName || '',
          Latitude: c.Latitude || '',
          Longitude: c.Longitude || '',
          status: c.status || 'offline'
        }));
      }
      
      globalRecentAlarmsList = (data.recent_alarms || []).map(a => ({
        Id: a.Id,
        CreatTime: a.CreatTime || '',
        Location: a.Location || a.AreaName || '未知位置',
        Picture: a.Picture || '',
        VideoUrl: a.VideoUrl || '',
        Confidence: a.Confidence || 0.95,
        CameraId: a.CameraId || 1,
        CameraName: a.CameraName || '',
        AreaName: a.AreaName || ''
      }));

      // Trigger map update if initialized
      updateMapMarkers();

      // Update Monthly Region Ranking TOP 5
      const rankingContainer = document.getElementById('rankingContainer');
      if (rankingContainer && data.monthly_ranking) {
        let rankingHtml = '';
        const maxRank = data.max_rank || 1;
        data.monthly_ranking.forEach(r => {
          const pct = Math.round(r.count / maxRank * 100);
          rankingHtml += `
          <div class="flex flex-col gap-1 text-xs">
            <div class="flex justify-between items-center text-slate-350">
              <span class="font-medium text-slate-200">${r.name}</span>
              <span class="font-mono text-cyan-400 font-bold">${r.count} 次</span>
            </div>
            <div class="w-full bg-slate-950/60 rounded-full h-1.5 overflow-hidden border border-slate-900">
              <div class="bg-gradient-to-r from-cyan-500 to-blue-500 h-full rounded-full" style="width: ${pct}%"></div>
            </div>
          </div>`;
        });
        if (data.monthly_ranking.length === 0) {
          rankingHtml = '<div class="text-center text-slate-500 py-4">暂无排行数据</div>';
        }
        rankingContainer.innerHTML = rankingHtml;
      }

      // Filter recent alarms locally based on query conditions
      let filteredAlarms = data.recent_alarms || [];
      if (filterStatus) {
        if (filterStatus === '1' || filterStatus === 'pending') {
          filteredAlarms = filteredAlarms.filter(a => a.Status === '1');
        } else if (filterStatus === 'processed') {
          filteredAlarms = filteredAlarms.filter(a => a.Status === '2' && a.OperateResult !== '误报无需处理' && a.OperateResult !== '漏报记录');
        } else if (filterStatus === 'false_alarm') {
          filteredAlarms = filteredAlarms.filter(a => a.Status === '2' && a.OperateResult === '误报无需处理');
        } else if (filterStatus === 'missed') {
          filteredAlarms = filteredAlarms.filter(a => a.Status === '2' && a.OperateResult === '漏报记录');
        } else {
          filteredAlarms = filteredAlarms.filter(a => a.Status === filterStatus);
        }
      }
      if (filterLocation) {
        filteredAlarms = filteredAlarms.filter(a => {
          const loc = (a.Location || a.AreaName || '').toLowerCase();
          return loc.includes(filterLocation);
        });
      }
      if (filterStart) {
        const startMs = new Date(filterStart).getTime();
        filteredAlarms = filteredAlarms.filter(a => {
          if (!a.CreatTime) return false;
          const alarmMs = new Date(a.CreatTime.replace(' ', 'T')).getTime();
          return alarmMs >= startMs;
        });
      }
      if (filterEnd) {
        const endMs = new Date(filterEnd).getTime();
        filteredAlarms = filteredAlarms.filter(a => {
          if (!a.CreatTime) return false;
          const alarmMs = new Date(a.CreatTime.replace(' ', 'T')).getTime();
          return alarmMs <= endMs;
        });
      }

      // Sort filteredAlarms locally
      filteredAlarms.sort((a, b) => {
        const getRank = (item) => {
          if (item.Status === '1') return 1;
          if (item.OperateResult === '误报无需处理') return 4;
          if (item.OperateResult === '漏报记录') return 3;
          return 2;
        };
        const rA = getRank(a);
        const rB = getRank(b);
        if (rA !== rB) return rA - rB;
        const cA = parseFloat(a.Confidence) || 0;
        const cB = parseFloat(b.Confidence) || 0;
        if (cA !== cB) return cB - cA;
        return new Date(b.CreatTime ? b.CreatTime.replace(' ', 'T') : 0) - new Date(a.CreatTime ? a.CreatTime.replace(' ', 'T') : 0);
      });

      // Update Left Timeline (recent_alarms)
      const timelineList = document.getElementById('timelineList');
      if (timelineList) {
        // Prevent layout flickering by only updating innerHTML if the content data actually changed
        const currentAlarmsStr = JSON.stringify(filteredAlarms.map(a => ({
          id: a.Id,
          status: a.Status,
          result: a.OperateResult,
          desc: a.Description,
          conf: a.Confidence,
          time: a.CreatTime,
          loc: a.Location
        })));
        if (timelineList.getAttribute('data-last-alarms') !== currentAlarmsStr) {
          timelineList.setAttribute('data-last-alarms', currentAlarmsStr);
          
          if (filteredAlarms.length === 0) {
            timelineList.innerHTML = '<div class="text-slate-600 text-center py-8">暂无报警记录</div>';
          } else {
            let timelineHtml = '';
            filteredAlarms.forEach(a => {
              const location = a.Location || a.AreaName || '未知位置';
              const timeStr = a.CreatTime || '--';
              const camera = a.CameraName || '摄像头1';
              const confBadge = a.Confidence ? '(' + (a.Confidence * 100).toFixed(1) + '%)' : '';
              
              const isSmoke = a.Description && a.Description.includes('烟雾') && !a.Description.includes('火焰');
              const typeName = isSmoke ? '烟雾报警' : '火焰预警';
              const typeColorClass = isSmoke ? 'text-sky-400' : 'text-rose-450';
              
              let dotBorderClass = isSmoke ? 'border-sky-500 shadow-[0_0_6px_#0ea5e9]' : 'border-rose-500 shadow-[0_0_6px_#f43f5e]';
              let dotBgClass = isSmoke ? 'bg-sky-500 text-sky-500' : 'bg-rose-500 text-rose-500';
              let statusBadge = isSmoke 
                ? '<span class="text-[10px] bg-sky-500/10 text-sky-400 border border-sky-500/20 px-1.5 py-0.5 rounded font-bold">待处理</span>'
                : '<span class="text-[10px] bg-rose-500/10 text-rose-400 border border-rose-500/20 px-1.5 py-0.5 rounded font-bold">待处理</span>';
              let pulseHtml = '';
              
              if (a.Status === '1') {
                pulseHtml = '<span class="radar-pulse"></span>';
              } else if (a.OperateResult === '误报无需处理') {
                dotBorderClass = 'border-emerald-500 shadow-[0_0_6px_#10b981]';
                dotBgClass = 'bg-emerald-500';
                statusBadge = '<span class="text-[10px] bg-emerald-500/15 text-emerald-450 border border-emerald-500/35 px-1.5 py-0.5 rounded font-bold">排除误报</span>';
              } else if (a.OperateResult === '漏报记录') {
                dotBorderClass = 'border-amber-500 shadow-[0_0_6px_#f59e0b]';
                dotBgClass = 'bg-amber-500 text-amber-500';
                pulseHtml = '<span class="radar-pulse"></span>';
                statusBadge = '<span class="text-[10px] bg-amber-500/15 text-amber-400 border border-amber-500/35 px-1.5 py-0.5 rounded font-bold animate-pulse">漏报记录</span>';
              } else {
                dotBorderClass = 'border-blue-500 shadow-[0_0_6px_#3b82f6]';
                dotBgClass = 'bg-blue-500';
                statusBadge = '<span class="text-[10px] bg-blue-500/15 text-blue-400 border border-blue-500/35 px-1.5 py-0.5 rounded font-bold">已处理</span>';
              }
              
              let leftBorderClass = isSmoke 
                ? 'border-l-sky-500 shadow-[0_0_8px_rgba(14,165,233,0.1)]'
                : 'border-l-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.1)]';
              if (a.Status !== '1') {
                if (a.OperateResult === '误报无需处理') {
                  leftBorderClass = 'border-l-emerald-500';
                } else if (a.OperateResult === '漏报记录') {
                  leftBorderClass = 'border-l-amber-500';
                } else {
                  leftBorderClass = 'border-l-blue-500';
                }
              }
              
              timelineHtml += `
              <div class="relative mb-2 timeline-item-anim">
                <span class="absolute -left-[21px] mt-1 w-2.5 h-2.5 rounded-full border bg-[#050c18] flex items-center justify-center ${dotBorderClass}">
                  <span class="w-1.5 h-1.5 rounded-full relative flex ${dotBgClass}">
                    ${pulseHtml}
                  </span>
                </span>
                <div class="glass-panel rounded-xl p-3 flex flex-col gap-1.5 cursor-pointer hover:border-cyan-500/20 hover:shadow-[0_0_15px_rgba(6,182,212,0.15)] transition duration-300 border-l-[3px] ${leftBorderClass}" onclick="showAlarmDetail(${a.Id})">
                  <div class="flex justify-between items-center">
                    <span class="${typeColorClass} font-semibold text-xs">${typeName} ${confBadge}</span>
                    ${statusBadge}
                  </div>
                  <div class="flex justify-between items-center text-xs text-slate-350 mt-1">
                    <span class="truncate max-w-[120px]">${location}</span>
                    <span class="text-cyan-400 font-mono text-[10px] truncate max-w-[80px]">${camera}</span>
                  </div>
                  <div class="flex justify-between items-center text-[10px] text-slate-500 mt-0.5">
                    <span>${timeStr}</span>
                    <span class="text-cyan-500/70">查看详情 →</span>
                  </div>
                </div>
              </div>`;
            });
            timelineList.innerHTML = timelineHtml;
          }
        }
      }

      // Update Middle Snapshots (recent_alarms with Picture) - 5-card scrollable layout
      const snapshotsContainer = document.getElementById('snapshotsContainer');
      if (snapshotsContainer) {
        const snapshots = (data.recent_alarms || []).filter(a => a.Picture);
        let snapshotsHtml = '';
        for (let i = 0; i < 5; i++) {
          if (i < snapshots.length) {
            const a = snapshots[i];
            const location = a.Location || a.AreaName || '--';
            const timeStr = a.CreatTime ? a.CreatTime.substring(11, 19) : '--';
            const conf = a.Confidence ? Math.round(a.Confidence * 100) : 95;
            
            const isSmoke = a.Description && a.Description.includes('烟雾') && !a.Description.includes('火焰');
            const typeLabel = isSmoke ? '烟雾报警' : '火灾预警';
            const badgeClass = isSmoke ? 'bg-sky-600/80' : 'bg-pink-600/80';
            const badgeLabel = isSmoke ? 'SMOKE' : 'FIRE';
            const alarmLabelHtml = isSmoke 
              ? `<span class="text-[9px] text-sky-400 font-bold bg-sky-500/10 border border-sky-500/20 px-1 py-0.5 rounded">烟雾报警</span>`
              : `<span class="text-[9px] text-rose-400 font-bold bg-rose-500/10 border border-rose-500/20 px-1 py-0.5 rounded">火灾报警</span>`;
            const cardBorderClass = isSmoke
              ? 'border-sky-500/50 shadow-[0_0_12px_rgba(14,165,233,0.15)]'
              : 'border-rose-500/50 shadow-[0_0_12px_rgba(244,63,94,0.15)]';

            snapshotsHtml += `
            <div class="w-[calc(20%-10px)] min-w-[160px] shrink-0 rounded-xl bg-slate-950/40 border ${cardBorderClass} p-2 flex flex-col gap-1.5 cursor-pointer hover:border-cyan-500/30 transition duration-300" onclick="showAlarmDetail(${a.Id})">
              <div class="relative aspect-video rounded-lg overflow-hidden border border-slate-800/60">
                <img src="${a.Picture}" class="w-full h-full object-cover">
                <span class="absolute top-1 left-1 ${isSmoke ? 'bg-sky-600/90' : 'bg-rose-600/90'} text-white text-[9px] font-bold px-1.5 py-0.5 rounded tracking-wider">${typeLabel}</span>
                <span class="absolute bottom-1 right-1 ${badgeClass} text-white text-[9px] font-bold px-1.5 py-0.5 rounded">${badgeLabel} ${conf}%</span>
              </div>
              <div class="flex justify-between items-center text-[10px] mt-0.5">
                <span class="text-orange-400 font-mono font-semibold">${timeStr}</span>
                <span class="text-slate-400 truncate max-w-[80px] font-medium">${location}</span>
              </div>
              <div class="flex justify-between items-center mt-1">
                <span class="text-[9px] text-slate-500">通道-0${a.CameraId || 1}</span>
                ${alarmLabelHtml}
              </div>
            </div>`;
          } else {
            snapshotsHtml += `
            <div class="w-[calc(20%-10px)] min-w-[160px] shrink-0 rounded-xl bg-slate-950/40 border border-slate-800/60 p-2 flex flex-col gap-1.5 opacity-60">
              <div class="relative aspect-video rounded-lg overflow-hidden border border-slate-800/45 bg-slate-900/60 flex items-center justify-center">
                <svg class="w-8 h-8 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                </svg>
              </div>
              <div class="flex justify-between items-center text-[10px] mt-0.5">
                <span class="text-slate-500 font-mono">--:--:--</span>
                <span class="text-slate-500 truncate max-w-[80px] font-medium">常规监控通道</span>
              </div>
              <div class="flex justify-between items-center mt-1">
                <span class="text-[9px] text-slate-500">通道-0${i + 1}</span>
                <span class="text-[9px] text-slate-500 font-semibold bg-slate-800/40 border border-slate-700/30 px-1 py-0.5 rounded">常规监控</span>
              </div>
            </div>`;
          }
        }
        snapshotsContainer.innerHTML = snapshotsHtml;
      }

      // Update new metric elements
      const statOnline = document.getElementById('statOnline');
      if (statOnline) statOnline.textContent = data.online_count || 0;
      const statOffline = document.getElementById('statOffline');
      if (statOffline) statOffline.textContent = data.offline_count || 0;
      const statAiRate = document.getElementById('statAiRate');
      if (statAiRate) {
        const rate = data.ai_rate !== undefined ? data.ai_rate : 100;
        statAiRate.innerHTML = rate + '<span class="text-sm">%</span>';
      }

      // Update Charts
      if (data.time_stats && trendChartInstance) {
        const xData = data.time_stats.map(item => item.date.substring(5)); // 'MM-DD'
        const yTotal = data.time_stats.map(item => item.total || 0);
        const yTrue = data.time_stats.map(item => item.true || 0);
        const yFalse = data.time_stats.map(item => item.false || 0);
        const yMissed = data.time_stats.map(item => item.missed || 0);
        const yPending = data.time_stats.map(item => item.pending || 0);
        trendChartInstance.setOption({
          xAxis: { data: xData },
          series: [
            { data: yTotal },
            { data: yTrue },
            { data: yFalse },
            { data: yMissed },
            { data: yPending }
          ]
        });
      }
      if (accuracyChartInstance) {
        accuracyChartInstance.setOption({
          series: [{
            data: [
              { value: data.true_count || 0, name: '确认火警', itemStyle: { color: '#f43f5e', shadowBlur: 8, shadowColor: 'rgba(244,63,94,0.4)' } },
              { value: data.false_count || 0, name: '误报记录', itemStyle: { color: '#10b981', shadowBlur: 8, shadowColor: 'rgba(16,185,129,0.4)' } },
              { value: data.missed_count || 0, name: '漏报记录', itemStyle: { color: '#f59e0b', shadowBlur: 8, shadowColor: 'rgba(245,158,11,0.4)' } },
              { value: data.pending_alarms || 0, name: '待处理', itemStyle: { color: '#06b6d4', shadowBlur: 8, shadowColor: 'rgba(6,182,212,0.4)' } }
            ]
          }]
        });
        // Update pie center total
        const total = (data.true_count || 0) + (data.false_count || 0) + (data.missed_count || 0) + (data.pending_alarms || 0);
        const pieTotal = document.getElementById('pieCenterTotal');
        if (pieTotal) pieTotal.textContent = total || data.total || 0;
        
        // Update percentages and counts legend
        const trueVal = data.true_count || 0;
        const falseVal = data.false_count || 0;
        const missedVal = data.missed_count || 0;
        const pendingVal = data.pending_alarms || 0;
        const totalVal = trueVal + falseVal + missedVal + pendingVal;
        
        const true_pct_clamped = totalVal > 0 ? Math.round(trueVal / totalVal * 100) : 0;
        const false_pct_clamped = totalVal > 0 ? Math.round(falseVal / totalVal * 100) : 0;
        const missed_pct_clamped = totalVal > 0 ? Math.round(missedVal / totalVal * 100) : 0;
        const pending_pct_clamped = totalVal > 0 ? (100 - true_pct_clamped - false_pct_clamped - missed_pct_clamped) : 0;
        
        const pctTrueEl = document.getElementById('pctTrue');
        if (pctTrueEl) pctTrueEl.textContent = true_pct_clamped + '%';
        const pctFalseEl = document.getElementById('pctFalse');
        if (pctFalseEl) pctFalseEl.textContent = false_pct_clamped + '%';
        const pctMissedEl = document.getElementById('pctMissed');
        if (pctMissedEl) pctMissedEl.textContent = missed_pct_clamped + '%';
        const pctPendingEl = document.getElementById('pctPending');
        if (pctPendingEl) pctPendingEl.textContent = Math.max(0, pending_pct_clamped) + '%';
        
        const trueEl = document.getElementById('statTrueCount');
        if (trueEl) trueEl.textContent = '(' + trueVal + '次)';
        const falseEl = document.getElementById('statFalseCount');
        if (falseEl) falseEl.textContent = '(' + falseVal + '次)';
        const missedEl = document.getElementById('statMissedCount');
        if (missedEl) missedEl.textContent = '(' + missedVal + '次)';
        const pendingEl = document.getElementById('statPendingCount');
        if (pendingEl) pendingEl.textContent = '(' + pendingVal + '次)';
      }
    })
    .catch(err => console.error('Error fetching stats:', err));
}

// Window resizing for ECharts responsive
window.addEventListener('resize', function() {
  if (typeof echarts !== 'undefined') {
    if (trendChartInstance) trendChartInstance.resize();
    if (accuracyChartInstance) accuracyChartInstance.resize();
  }
});

// Initialization
document.addEventListener("DOMContentLoaded", function() {
  // Safe ECharts configuration & initialization
  if (typeof echarts !== 'undefined') {
    const chartTextColor = '#94a3b8';
    const chartLineColor = 'rgba(51, 65, 85, 0.3)';

    accuracyOption = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} 次 ({d}%)',
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        borderColor: 'rgba(34, 211, 238, 0.3)',
        textStyle: { color: '#f1f5f9', fontSize: 10 },
        borderWidth: 1
      },
      legend: { show: false },
      series: [
        {
          name: '警情比例',
          type: 'pie',
          radius: ['55%', '78%'],
          center: ['50%', '50%'],
          avoidLabelOverlap: false,
          label: { show: false },
          emphasis: {
            label: { show: true, fontSize: 11, fontWeight: 'bold', color: '#f1f5f9' }
          },
          labelLine: { show: false },
          itemStyle: { borderColor: 'rgba(5,12,24,0.8)', borderWidth: 3, borderRadius: 3 },
          data: [
            { value: {{ true_count }}, name: '确认火警', itemStyle: { color: '#f43f5e', shadowBlur: 8, shadowColor: 'rgba(244,63,94,0.4)' } },
            { value: {{ false_count }}, name: '误报记录', itemStyle: { color: '#10b981', shadowBlur: 8, shadowColor: 'rgba(16,185,129,0.4)' } },
            { value: {{ missed_count }}, name: '漏报记录', itemStyle: { color: '#f59e0b', shadowBlur: 8, shadowColor: 'rgba(245,158,11,0.4)' } },
            { value: {{ pending_alarms }}, name: '待处理', itemStyle: { color: '#06b6d4', shadowBlur: 8, shadowColor: 'rgba(6,182,212,0.4)' } }
          ]
        }
      ]
    };

    trendOption = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        borderColor: 'rgba(34, 211, 238, 0.5)',
        textStyle: { color: '#f1f5f9', fontSize: 11 },
        borderWidth: 1,
        borderRadius: 8,
        shadowColor: 'rgba(0, 0, 0, 0.5)',
        shadowBlur: 10
      },
      legend: {
        data: ['全部报警', '确认火警', '误报记录', '漏报记录', '待处理'],
        textStyle: { color: '#94a3b8', fontSize: 10 },
        top: '0%',
        right: '2%',
        itemWidth: 12,
        itemHeight: 8
      },
      grid: {
        top: '22%',
        left: '2%',
        right: '2%',
        bottom: '2%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: [],
        axisLine: { lineStyle: { color: chartLineColor } },
        axisLabel: { color: '#94a3b8', fontSize: 10, fontWeight: 'medium' },
        splitLine: { show: false }
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisLabel: { color: '#94a3b8', fontSize: 10, fontWeight: 'medium' },
        splitLine: { lineStyle: { color: chartLineColor, type: 'dashed' } }
      },
      series: [
        {
          name: '全部报警',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          showSymbol: false,
          itemStyle: { color: '#22d3ee' },
          lineStyle: { width: 2 },
          data: []
        },
        {
          name: '确认火警',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          showSymbol: false,
          itemStyle: { color: '#f43f5e' },
          lineStyle: { width: 2 },
          data: []
        },
        {
          name: '误报记录',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          showSymbol: false,
          itemStyle: { color: '#10b981' },
          lineStyle: { width: 2 },
          data: []
        },
        {
          name: '漏报记录',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          showSymbol: false,
          itemStyle: { color: '#f59e0b' },
          lineStyle: { width: 2 },
          data: []
        },
        {
          name: '待处理',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          showSymbol: false,
          itemStyle: { color: '#06b6d4' },
          lineStyle: { width: 2 },
          data: []
        }
      ]
    };

    const trendDom = document.getElementById('trendChart');
    if (trendDom) {
      trendChartInstance = echarts.init(trendDom);
      trendChartInstance.setOption(trendOption);
    }
    const accuracyDom = document.getElementById('accuracyChart');
    if (accuracyDom) {
      accuracyChartInstance = echarts.init(accuracyDom);
      accuracyChartInstance.setOption(accuracyOption);
    }
  }
  
  // Safe initialization of Scanner and WebSocket stream
  const scanIntervalInput = document.getElementById('scanIntervalInput');
  if (scanIntervalInput) {
    scanIntervalInput.addEventListener('change', initScanner);
  }
  
  initScanner();
  wsConnect();
  
  fetchRealtimeData();
  setInterval(fetchRealtimeData, 2000);

  // Intercept alarm form submit to prevent page reload and protect stream
  const modalForm = document.getElementById('modalProcessForm');
  if (modalForm) {
    modalForm.addEventListener('submit', function(e) {
      e.preventDefault();
      const form = this;
      const formData = new FormData(form);
      fetch(form.action, {
        method: 'POST',
        body: formData
      })
      .then(res => {
        closeAlarmDetail();
        fetchRealtimeData();
      })
      .catch(err => {
        console.error('Failed to submit alarm process form:', err);
        closeAlarmDetail();
        fetchRealtimeData();
      });
    });
  }
});
</script>

<!-- Scan Settings Modal -->
<div id="scanModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/85 backdrop-blur-sm p-4 animate-fade-in">
  <div class="glass-panel w-full max-w-sm rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col animate-scale-in">
    <div class="px-5 py-3 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
        扫描设置
      </h3>
      <button onclick="closeScanModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <div class="p-5 flex flex-col gap-4 text-xs">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">扫描 IP 地址</label>
        <input type="text" id="scanHost" value="0.0.0.0" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-cyan-400 text-center font-mono focus:outline-none focus:border-cyan-500/50 transition">
      </div>
      <div class="grid grid-cols-2 gap-3">
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">起始端口</label>
          <input type="number" id="scanStart" value="9991" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-cyan-400 text-center font-mono focus:outline-none focus:border-cyan-500/50 transition [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none">
        </div>
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">结束端口</label>
          <input type="number" id="scanEnd" value="9994" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-cyan-400 text-center font-mono focus:outline-none focus:border-cyan-500/50 transition [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none">
        </div>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">扫描间隔 (秒)</label>
        <input type="number" id="scanIntervalInput" value="1" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-cyan-400 text-center font-mono focus:outline-none focus:border-cyan-500/50 transition [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none">
      </div>
      <div class="flex gap-2.5 mt-1">
        <button onclick="triggerDefaultScan();closeScanModal();" class="flex-1 bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-400 py-2 rounded-lg font-semibold transition active:scale-95">AUTO 扫描</button>
        <button onclick="triggerManualScan();closeScanModal();" class="flex-1 bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/30 text-cyan-400 py-2 rounded-lg font-semibold transition active:scale-95">MANUAL 扫描</button>
      </div>
    </div>
  </div>
</div>

<script>
function openScanModal() { document.getElementById('scanModal').classList.remove('hidden'); }
function closeScanModal() { document.getElementById('scanModal').classList.add('hidden'); }
</script>

<!-- Alarm Detail Modal -->
<div id="detailModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4 animate-fade-in">
  <div class="glass-panel w-full max-w-2xl rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col max-h-[90vh] animate-scale-in">
    <!-- Modal Header -->
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
        预警事件详情
      </h3>
      <button onclick="closeAlarmDetail()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    
    <!-- Modal Body -->
    <div class="p-6 overflow-y-auto flex flex-col md:flex-row gap-6 text-xs scrollbar-thin">
      <!-- Left Side: Image & Video -->
      <div class="flex-1 flex flex-col gap-4 animate-fade-in">
        <div class="flex flex-col gap-1.5">
          <span class="text-slate-400 font-semibold">现场图片</span>
          <div class="aspect-video bg-slate-950/50 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center relative">
            <img id="modalAlarmImage" src="" class="w-full h-full object-cover hidden">
            <div id="modalAlarmNoImage" class="text-slate-500 flex flex-col items-center gap-1.5 py-8">
              <svg class="w-8 h-8 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              <span>暂无现场图片</span>
            </div>
          </div>
        </div>
        <div class="flex flex-col gap-1.5">
          <span class="text-slate-400 font-semibold">视频回放</span>
          <div class="aspect-video bg-slate-950/50 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center relative">
            <video id="modalAlarmVideo" src="" class="w-full h-full object-cover hidden" controls autoplay muted></video>
            <div id="modalAlarmNoVideo" class="text-slate-500 flex flex-col items-center gap-1.5 py-8">
              <svg class="w-8 h-8 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
              </svg>
              <span>暂无录像视频</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Right Side: Info & Form -->
      <div class="flex-1 flex flex-col gap-4">
        <!-- Event Metadata -->
        <div class="bg-slate-905/30 border border-slate-800 rounded-xl p-4 flex flex-col gap-2.5">
          <div class="flex justify-between items-center border-b border-slate-800/60 pb-2">
            <span class="text-slate-400">发生地点</span>
            <span class="font-semibold text-slate-200" id="modalAlarmLocation">--</span>
          </div>
          <div class="flex justify-between items-center border-b border-slate-800/60 pb-2">
            <span class="text-slate-400">预警时间</span>
            <span class="font-mono text-slate-300" id="modalAlarmTime">--</span>
          </div>
          <div class="flex justify-between items-center border-b border-slate-800/60 pb-2">
            <span class="text-slate-400">预警类型</span>
            <span id="modalAlarmType" class="font-semibold">--</span>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-slate-400">预警状态</span>
            <span id="modalAlarmStatus" class="font-bold">--</span>
          </div>
        </div>
        
        <!-- Processing Form (For Status = '1') -->
        {% if user.RoleName == '审核人' %}
        <div id="modalProcessForm" class="hidden flex flex-col gap-3 bg-slate-900/20 border border-rose-500/20 rounded-xl p-4 text-center text-rose-400 font-semibold shadow-inner">
          🛡️ 您的角色为审核人，无权处理报警事件。
          <button type="button" onclick="closeAlarmDetail()" class="w-full mt-2 bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-lg font-semibold transition active:scale-95 border border-slate-700">关闭详情</button>
        </div>
        {% else %}
        <form id="modalProcessForm" method="post" action="" class="hidden flex flex-col gap-3">
          <div class="flex flex-col gap-1">
            <label class="text-slate-400 font-medium">紧急程度</label>
            <select name="UrgencyDegree" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
              <option value="普通">普通</option>
              <option value="紧急">紧急</option>
              <option value="特急">特急</option>
            </select>
          </div>
          <div class="flex flex-col gap-1">
            <label class="text-slate-400 font-medium">处理结果</label>
            <select name="OperateResult" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
              <option value="火灾已确认并报警">火灾已确认并报警</option>
              <option value="误报无需处理">误报无需处理</option>
              <option value="漏报记录">漏报记录</option>
              <option value="其它已处理">其它已处理</option>
            </select>
          </div>
          <div class="flex flex-col gap-1">
            <label class="text-slate-400 font-medium">处理备注 / 描述</label>
            <textarea name="Description" id="processDescription" rows="2" placeholder="请输入处理备注信息..." class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 transition resize-none"></textarea>
          </div>
          <div class="flex gap-2.5 mt-2">
            <button type="submit" class="flex-1 bg-rose-600/90 hover:bg-rose-600 text-white py-2 rounded-lg font-bold transition active:scale-95 shadow-md shadow-rose-950/50">确认处理</button>
            <button type="button" onclick="closeAlarmDetail()" class="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-lg font-semibold transition active:scale-95 border border-slate-700">暂不处理</button>
          </div>
        </form>
        {% endif %}
        
        <!-- Processed Information (For Status = '2' or '3') -->
        <div id="modalProcessedInfo" class="hidden flex flex-col gap-3 bg-slate-900/20 border border-slate-800/40 rounded-xl p-3.5 text-[11px] text-slate-300">
          <div class="flex justify-between border-b border-slate-800/60 pb-1.5">
            <span class="text-slate-400">处理人</span>
            <span class="font-medium text-slate-200" id="modalInfoOperator">--</span>
          </div>
          <div class="flex justify-between border-b border-slate-800/60 pb-1.5">
            <span class="text-slate-400">处理时间</span>
            <span class="font-mono text-slate-200" id="modalInfoTime">--</span>
          </div>
          <div class="flex justify-between border-b border-slate-800/60 pb-1.5">
            <span class="text-slate-400">处理结果</span>
            <span class="font-medium text-orange-400" id="modalInfoResult">--</span>
          </div>
          <div class="flex justify-between border-b border-slate-800/60 pb-1.5">
            <span class="text-slate-400">紧急程度</span>
            <span class="font-semibold text-rose-450" id="modalInfoUrgency">--</span>
          </div>
          <div class="flex flex-col gap-1">
            <span class="text-slate-400">备注描述</span>
            <p class="text-slate-350 leading-relaxed bg-slate-950/40 rounded-lg p-2 border border-slate-850" id="modalInfoDesc">--</p>
          </div>
          
          <!-- Audit Section for Status '2' and authorized users -->
          {% if user.RoleName in ['超级管理员', '审核人'] %}
          <div id="modalAuditActions" class="hidden flex gap-2 mt-2">
            <a id="modalAuditApproveBtn" href="" class="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white text-center py-2 rounded-lg font-semibold transition active:scale-95 shadow-md shadow-emerald-950/30">审核通过</a>
            <a id="modalAuditRejectBtn" href="" class="flex-1 bg-rose-600 hover:bg-rose-500 text-white text-center py-2 rounded-lg font-semibold transition active:scale-95 shadow-md shadow-rose-950/30">驳回</a>
          </div>
          {% endif %}
          
          <button type="button" onclick="closeAlarmDetail()" class="w-full bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-lg font-semibold transition mt-2">关闭</button>
        </div>
      </div>
    </div>
  </div>
</div>
</body>
</html>
"""

def make_admin_template(title, content_html, active_menu):
    """构建管理后台通用页面模板：拼接公共导航栏 + 左侧边栏 + 主体内容区域。
    
    根据 active_menu 参数高亮当前侧边栏菜单项。非超级管理员用户不显示资源管理和系统设置菜单。
    
    Args:
        title (str): 页面标题（显示在浏览器标签页）
        content_html (str): 主体内容区域的 HTML 代码
        active_menu (str): 当前活跃菜单标识（如 'device', 'alarm', 'config' 等）
        
    Returns:
        str: 完整的 HTML 页面字符串
    """
    # Build sidebar active item with vertical cyan bar
    def sidebar_link(href, icon, label, menu_name):
        if active_menu == menu_name:
            return f'<a href="{href}" class="relative flex items-center gap-2.5 px-3 py-2.5 rounded-r-lg text-xs font-semibold transition bg-cyan-500/8 text-cyan-400 border-cyan-500/20"><span class="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 bg-cyan-400 rounded-r-full shadow-[0_0_8px_#22d3ee]"></span><span class="ml-1">{icon}</span> {label}</a>'
        return f'<a href="{href}" class="flex items-center gap-2.5 px-4 py-2.5 rounded-r-lg text-xs font-semibold transition text-slate-400 hover:text-slate-200 hover:bg-slate-900/30">{icon} {label}</a>'

    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>""" + title + """ - 视频AI智能识别平台</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://lib.baomitu.com/jquery/3.6.0/jquery.min.js"></script>
  <style>
    .scrollbar-thin::-webkit-scrollbar { width: 6px; height: 6px; }
    .scrollbar-thin::-webkit-scrollbar-track { background: transparent; }
    .scrollbar-thin::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
    .scrollbar-thin::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }
    .glass-panel {
      background: rgba(15, 23, 42, 0.45);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes scaleIn {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }
    .animate-fade-in {
      animation: fadeIn 0.2s ease-out forwards;
    }
    .animate-scale-in {
      animation: scaleIn 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }
    body { background-color: #050c18; }
  </style>
</head>
<body class="bg-gradient-to-tr from-[#030712] via-[#091124] to-[#030712] text-slate-100 min-h-screen flex flex-col font-sans">
""" + BASE_NAV + """
<div class="flex flex-1 min-h-[calc(100vh-56px)]">
  <!-- Sidebar -->
  <aside class="w-60 bg-[#060e1a]/80 border-r border-slate-800/40 py-5 shrink-0 flex flex-col gap-5 backdrop-blur-md">
    {% if user.RoleName == '超级管理员' %}
    <div class="flex flex-col gap-0.5">
      <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-4 mb-1">资源管理</div>
      """ + sidebar_link('/admin/device', '💻', 'AI分析盒管理', 'device') + """
      """ + sidebar_link('/admin/camera', '📷', '摄像头管理', 'camera') + """
    </div>
    {% endif %}

    <div class="flex flex-col gap-0.5">
      <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-4 mb-1">事件处理</div>
      """ + sidebar_link('/admin/alarm', '🚨', '报警事件', 'alarm') + """
      """ + sidebar_link('/admin/audit', '🛡️', '事件处理审核', 'audit') + """
    </div>

    {% if user.RoleName == '超级管理员' %}
    <div class="flex flex-col gap-0.5">
      <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-4 mb-1">系统设置</div>
      """ + sidebar_link('/admin/config', '⚙️', '系统参数配置', 'config') + """
      """ + sidebar_link('/admin/branch', '🏢', '部门/机构管理', 'branch') + """
      """ + sidebar_link('/admin/user', '👤', '用户账户管理', 'user') + """
      """ + sidebar_link('/admin/role', '🔑', '角色权限管理', 'role') + """
      """ + sidebar_link('/admin/dictionary', '📖', '数据字典项', 'dictionary') + """
    </div>

    <div class="flex flex-col gap-0.5">
      <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-4 mb-1">故障与日志</div>
      """ + sidebar_link('/admin/camera_error', '⚠️', '摄像头故障', 'camera_error') + """
      """ + sidebar_link('/admin/device_error', '📦', 'AI分析盒故障', 'device_error') + """
      """ + sidebar_link('/admin/log/access', '🔒', '访问安全日志', 'access_log') + """
      """ + sidebar_link('/admin/log/operate', '📝', '业务操作日志', 'operate_log') + """
    </div>
    {% endif %}
  </aside>

  <!-- Main Content -->
  <main class="flex-1 p-6 overflow-y-auto scrollbar-thin flex flex-col gap-5">
    <!-- Welcome Banner -->
    <div class="bg-emerald-500/5 border border-emerald-500/15 rounded-xl px-5 py-3 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 bg-emerald-500/10 rounded-lg flex items-center justify-center text-emerald-400 text-sm">👋</div>
        <span class="text-sm text-emerald-300 font-semibold">欢迎回来，<span class="text-emerald-400">{{ user.Name }}</span>！</span>
      </div>
      <span class="text-[10px] text-slate-500">视频 AI 智能识别及预警管理平台</span>
    </div>
    """ + content_html + """
  </main>
</div>
</body>
</html>
"""

# 系统参数配置页面模板（超级管理员可见，用于修改检测阈值、视频参数、心跳间隔等全局设置）
CONFIG_TEMPLATE = make_admin_template("系统参数配置", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-cyan-500 rounded-full shadow-[0_0_8px_#06b6d4]"></span>
      系统参数配置
    </h2>
  </div>

  {% with msgs=get_flashed_messages() %}
  {% if msgs %}
  <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-xs font-semibold">
    {{ msgs[0] }}
  </div>
  {% endif %}
  {% endwith %}

  <div class="glass-panel rounded-2xl p-6 shadow-xl max-w-3xl">
    <form method="post" class="flex flex-col gap-5">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div class="flex flex-col gap-1.5">
          <label class="text-xs font-bold text-slate-400">站点名称</label>
          <input type="text" name="Name" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" value="{{ site.Name }}">
        </div>
        
        <div class="flex flex-col gap-1.5">
          <label class="text-xs font-bold text-slate-400">烟雾检测conf阈值</label>
          <input type="number" step="0.01" name="thresh" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" value="{{ site.thresh }}">
          <span class="text-[10px] text-slate-500">阈值越高越不容易报警，阈值越低越容易报警</span>
        </div>
        
        <div class="flex flex-col gap-1.5">
          <label class="text-xs font-bold text-slate-400">图片/视频长 (宽度)</label>
          <input type="number" name="width" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" value="{{ site.width }}">
        </div>
        
        <div class="flex flex-col gap-1.5">
          <label class="text-xs font-bold text-slate-400">图片/视频宽 (高度)</label>
          <input type="number" name="height" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" value="{{ site.height }}">
        </div>
        
        <div class="flex flex-col gap-1.5">
          <label class="text-xs font-bold text-slate-400">视频秒数</label>
          <input type="number" name="video_times" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" value="{{ site.video_times }}">
        </div>
        
        <div class="flex flex-col gap-1.5">
          <label class="text-xs font-bold text-slate-400">连接心跳时间 (小时)</label>
          <input type="number" name="heartBeat" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" value="{{ site.heartBeat }}">
        </div>
        
        <div class="flex flex-col gap-1.5 md:col-span-2">
          <label class="text-xs font-bold text-slate-400">网络异常误差 (分钟)</label>
          <input type="number" name="exception_times" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" value="{{ site.exception_times }}">
        </div>
      </div>
      
      <div class="flex justify-end mt-2">
        <button type="submit" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-6 py-2 rounded-lg font-semibold transition active:scale-95">保存配置</button>
      </div>
    </form>
  </div>
</div>
""", "config")

# 部门/机构管理页面模板（支持树形结构、新增、编辑、删除操作）
BRANCH_TEMPLATE = make_admin_template("部门管理", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-cyan-500 rounded-full shadow-[0_0_8px_#06b6d4]"></span>
      部门/机构管理
    </h2>
    <button onclick="openAddModal()" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition active:scale-95 flex items-center gap-1.5 text-xs">
      <span>➕</span> 新增部门
    </button>
  </div>

  {% with msgs=get_flashed_messages() %}
  {% if msgs %}
  <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-xs font-semibold">
    {{ msgs[0] }}
  </div>
  {% endif %}
  {% endwith %}

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">部门名称</th>
          <th class="px-4 py-3">上级部门</th>
          <th class="px-4 py-3">备注</th>
          <th class="px-4 py-3">操作</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for b in branches %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ b.Id }}</td>
          <td class="px-4 py-3.5 text-slate-200 font-medium">{{ b.Name }}</td>
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ b.ParentId }}</td>
          <td class="px-4 py-3.5 text-slate-400">{{ b.Remark or '--' }}</td>
          <td class="px-4 py-3.5 flex items-center gap-2">
            <button class="bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95" onclick="editBranch({{ b.Id }},'{{ b.Name }}',{{ b.ParentId }},'{{ b.Remark or '' }}')">修改</button>
            <a href="/admin/branch/delete/{{ b.Id }}" class="bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95" onclick="return confirm('确认删除?')">删除</a>
          </td>
        </tr>
        {% else %}
        <tr>
          <td colspan="5" class="px-4 py-8 text-center text-slate-500">暂无部门数据</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- Add Modal -->
<div id="addModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-md rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
        新增部门
      </h3>
      <button onclick="closeAddModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" action="/admin/branch/add" class="p-6 flex flex-col gap-4 text-xs">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">部门名称</label>
        <input type="text" name="Name" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">上级部门ID</label>
        <input type="number" name="ParentId" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" value="0">
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">备注</label>
        <textarea name="Remark" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition resize-none" rows="2"></textarea>
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeAddModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition">取消</button>
        <button type="submit" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition">保存</button>
      </div>
    </form>
  </div>
</div>

<!-- Edit Modal -->
<div id="editModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-md rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
        修改部门
      </h3>
      <button onclick="closeEditModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" id="editForm" class="p-6 flex flex-col gap-4 text-xs">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">部门名称</label>
        <input type="text" name="Name" id="eName" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">上级部门ID</label>
        <input type="number" name="ParentId" id="eParent" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">备注</label>
        <textarea name="Remark" id="eRemark" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition resize-none" rows="2"></textarea>
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeEditModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition">取消</button>
        <button type="submit" class="bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-400 px-4 py-2 rounded-lg font-semibold transition">保存</button>
      </div>
    </form>
  </div>
</div>

<script>
function openAddModal() { document.getElementById('addModal').classList.remove('hidden'); }
function closeAddModal() { document.getElementById('addModal').classList.add('hidden'); }
function closeEditModal() { document.getElementById('editModal').classList.add('hidden'); }
function editBranch(id,name,parent,remark){
  document.getElementById('editForm').action='/admin/branch/edit/'+id;
  document.getElementById('eName').value=name;
  document.getElementById('eParent').value=parent;
  document.getElementById('eRemark').value=remark;
  document.getElementById('editModal').classList.remove('hidden');
}
</script>
""", "branch")

# 用户账户管理页面模板（支持新增、编辑、软删除，含账号/姓名/部门/区域/角色字段）
USER_TEMPLATE = make_admin_template("用户管理", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-cyan-500 rounded-full shadow-[0_0_8px_#06b6d4]"></span>
      用户账户管理
    </h2>
    <button onclick="openAddModal()" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition active:scale-95 flex items-center gap-1.5 text-xs">
      <span>➕</span> 新增用户
    </button>
  </div>

  {% with msgs=get_flashed_messages() %}
  {% if msgs %}
  <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-xs font-semibold">
    {{ msgs[0] }}
  </div>
  {% endif %}
  {% endwith %}

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">账号</th>
          <th class="px-4 py-3">姓名</th>
          <th class="px-4 py-3">部门</th>
          <th class="px-4 py-3">区域</th>
          <th class="px-4 py-3">角色</th>
          <th class="px-4 py-3">操作</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for u in users %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ u.Id }}</td>
          <td class="px-4 py-3.5 text-slate-200 font-medium">{{ u.Account }}</td>
          <td class="px-4 py-3.5 text-slate-200">{{ u.Name }}</td>
          <td class="px-4 py-3.5 text-slate-400">{{ u.BranchName or '--' }}</td>
          <td class="px-4 py-3.5 text-slate-400">{{ u.AreaName or '--' }}</td>
          <td class="px-4 py-3.5"><span class="px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-[10px]">{{ u.RoleName or '--' }}</span></td>
          <td class="px-4 py-3.5 flex items-center gap-2">
            <button class="bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95" onclick="editUser({{ u.Id }},'{{ u.Account }}','{{ u.Name }}',{{ u.AreaId or 1 }},{{ u.BranchId or 1 }},'{{ u.RoleName }}')">修改</button>
            <a href="/admin/user/delete/{{ u.Id }}" class="bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95" onclick="return confirm('确认删除?')">删除</a>
          </td>
        </tr>
        {% else %}
        <tr>
          <td colspan="7" class="px-4 py-8 text-center text-slate-500">暂无用户数据</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- Add Modal -->
<div id="addModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-md rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
        新增用户
      </h3>
      <button onclick="closeAddModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" action="/admin/user/add" class="p-6 flex flex-col gap-4 text-xs">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">账号</label>
        <input type="text" name="Account" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">姓名</label>
        <input type="text" name="Name" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">密码</label>
        <input type="password" name="Password" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">区域</label>
        <select name="AreaId" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
          {% for a in areas %}
          <option value="{{ a.Id }}">{{ a.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">部门</label>
        <select name="BranchId" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
          {% for b in branches %}
          <option value="{{ b.Id }}">{{ b.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">角色</label>
        <select name="RoleId" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
          {% for r in roles %}
          <option value="{{ r.Id }}">{{ r.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeAddModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition">取消</button>
        <button type="submit" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition">保存</button>
      </div>
    </form>
  </div>
</div>

<!-- Edit Modal -->
<div id="editModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-md rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
        修改用户
      </h3>
      <button onclick="closeEditModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" id="editForm" class="p-6 flex flex-col gap-4 text-xs">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">账号</label>
        <input type="text" name="Account" id="eAcc" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">姓名</label>
        <input type="text" name="Name" id="eName" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">新密码 (留空不修改)</label>
        <input type="password" name="Password" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">区域</label>
        <select name="AreaId" id="eArea" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
          {% for a in areas %}
          <option value="{{ a.Id }}">{{ a.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">部门</label>
        <select name="BranchId" id="eBranch" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
          {% for b in branches %}
          <option value="{{ b.Id }}">{{ b.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">角色</label>
        <select name="RoleId" id="eRole" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
          {% for r in roles %}
          <option value="{{ r.Id }}">{{ r.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeEditModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition">取消</button>
        <button type="submit" class="bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-400 px-4 py-2 rounded-lg font-semibold transition">保存</button>
      </div>
    </form>
  </div>
</div>

<script>
function openAddModal() { document.getElementById('addModal').classList.remove('hidden'); }
function closeAddModal() { document.getElementById('addModal').classList.add('hidden'); }
function closeEditModal() { document.getElementById('editModal').classList.add('hidden'); }
function editUser(id,acc,name,area,branch,roleName){
  document.getElementById('editForm').action='/admin/user/edit/'+id;
  document.getElementById('eAcc').value=acc;
  document.getElementById('eName').value=name;
  document.getElementById('eArea').value=area;
  document.getElementById('eBranch').value=branch;
  Array.from(document.getElementById('eRole').options).forEach(function(o){
    o.selected = o.text === roleName;
  });
  document.getElementById('editModal').classList.remove('hidden');
}
</script>
""", "user")

# 角色权限管理页面模板（支持多选权限分配，权限列表包括系统配置、部门、用户、角色、设备、摄像头、报警、审核、日志、仪表盘、字典）
ROLE_TEMPLATE = make_admin_template("角色管理", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-cyan-500 rounded-full shadow-[0_0_8px_#06b6d4]"></span>
      角色权限管理
    </h2>
    <button onclick="openAddModal()" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition active:scale-95 flex items-center gap-1.5 text-xs">
      <span>➕</span> 新增角色
    </button>
  </div>

  {% with msgs=get_flashed_messages() %}
  {% if msgs %}
  <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-xs font-semibold">
    {{ msgs[0] }}
  </div>
  {% endif %}
  {% endwith %}

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">角色名</th>
          <th class="px-4 py-3">描述</th>
          <th class="px-4 py-3">操作</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for r in roles %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ r.Id }}</td>
          <td class="px-4 py-3.5 text-slate-200 font-medium">{{ r.Name }}</td>
          <td class="px-4 py-3.5 text-slate-400">{{ r.Description or '--' }}</td>
          <td class="px-4 py-3.5">
            <a href="/admin/role/delete/{{ r.Id }}" class="bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95" onclick="return confirm('确认删除?')">删除</a>
          </td>
        </tr>
        {% else %}
        <tr>
          <td colspan="4" class="px-4 py-8 text-center text-slate-500">暂无角色数据</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- Add Modal -->
<div id="addModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-lg rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
        新增角色
      </h3>
      <button onclick="closeAddModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" action="/admin/role/add" class="p-6 flex flex-col gap-4 text-xs">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">角色名</label>
        <input type="text" name="Name" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">描述</label>
        <textarea name="Description" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition resize-none" rows="2"></textarea>
      </div>
      <div class="flex flex-col gap-2">
        <label class="text-slate-400 font-medium">权限分配</label>
        <div class="grid grid-cols-2 gap-2 bg-slate-950/50 border border-slate-900 rounded-xl p-3.5">
          {% set all_auths = ['system_config','department','user','role','device','camera','alarm','audit','log','dashboard','dictionary'] %}
          {% set auth_names = {'system_config':'系统配置','department':'部门管理','user':'用户管理','role':'角色管理','device':'AI分析盒','camera':'摄像头','alarm':'报警事件','audit':'事件审核','log':'日志管理','dashboard':'数据大屏','dictionary':'数据字典'} %}
          {% for a in all_auths %}
          <label class="flex items-center gap-2 text-slate-300 hover:text-slate-100 transition cursor-pointer select-none">
            <input type="checkbox" name="authorities" value="{{ a }}" class="rounded border-slate-800 bg-slate-950 text-cyan-500 focus:ring-cyan-500/50">
            <span>{{ auth_names.get(a, a) }}</span>
          </label>
          {% endfor %}
        </div>
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeAddModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition">取消</button>
        <button type="submit" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition">保存</button>
      </div>
    </form>
  </div>
</div>

<script>
function openAddModal() { document.getElementById('addModal').classList.remove('hidden'); }
function closeAddModal() { document.getElementById('addModal').classList.add('hidden'); }
</script>
""", "role")

# 数据字典管理页面模板（Key-Value 键值对，用于下拉选项等可配置枚举值的维护）
DICT_TEMPLATE = make_admin_template("数据字典", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-cyan-500 rounded-full shadow-[0_0_8px_#06b6d4]"></span>
      数据字典管理
    </h2>
    <button onclick="openAddModal()" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition active:scale-95 flex items-center gap-1.5 text-xs">
      <span>➕</span> 新增字典项
    </button>
  </div>

  {% with msgs=get_flashed_messages() %}
  {% if msgs %}
  <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-xs font-semibold">
    {{ msgs[0] }}
  </div>
  {% endif %}
  {% endwith %}

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">Key (类别键)</th>
          <th class="px-4 py-3">Value (字典值)</th>
          <th class="px-4 py-3">备注</th>
          <th class="px-4 py-3">操作</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for it in items %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ it.Id }}</td>
          <td class="px-4 py-3.5 text-slate-200 font-medium font-mono">{{ it.Key }}</td>
          <td class="px-4 py-3.5 text-slate-200">{{ it.Value }}</td>
          <td class="px-4 py-3.5 text-slate-400">{{ it.Remark or '--' }}</td>
          <td class="px-4 py-3.5">
            <a href="/admin/dictionary/delete/{{ it.Id }}" class="bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95" onclick="return confirm('确认删除?')">删除</a>
          </td>
        </tr>
        {% else %}
        <tr>
          <td colspan="5" class="px-4 py-8 text-center text-slate-500">暂无数据字典项</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- Add Modal -->
<div id="addModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-md rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
        新增字典项
      </h3>
      <button onclick="closeAddModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" action="/admin/dictionary/add" class="p-6 flex flex-col gap-4 text-xs">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">Key</label>
        <input type="text" name="Key" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">Value</label>
        <input type="text" name="Value" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition" required>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">备注</label>
        <input type="text" name="Remark" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeAddModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition">取消</button>
        <button type="submit" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition">保存</button>
      </div>
    </form>
  </div>
</div>

<script>
function openAddModal() { document.getElementById('addModal').classList.remove('hidden'); }
function closeAddModal() { document.getElementById('addModal').classList.add('hidden'); }
</script>
""", "dictionary")

# AI分析盒（边缘计算设备）管理页面模板（维护MAC、位置、区域、模型版本等信息）
DEVICE_TEMPLATE = make_admin_template("AI分析盒管理", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-3">
      <div class="w-9 h-9 bg-cyan-500/10 border border-cyan-500/20 rounded-xl flex items-center justify-center text-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.15)]">
        💻
      </div>
      AI分析盒管理
    </h2>
    <button onclick="openAddModal()" class="bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 text-cyan-400 px-4 py-2.5 rounded-lg font-semibold transition active:scale-95 flex items-center gap-1.5 text-xs">
      <span>➕</span> 新增AI分析盒
    </button>
  </div>

  {% with msgs=get_flashed_messages() %}
  {% if msgs %}
  <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-sm font-semibold">
    {{ msgs[0] }}
  </div>
  {% endif %}
  {% endwith %}

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-sm">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3.5">ID</th>
          <th class="px-4 py-3.5">MAC地址</th>
          <th class="px-4 py-3.5">物理位置</th>
          <th class="px-4 py-3.5">所属区域</th>
          <th class="px-4 py-3.5">在线状态</th>
          <th class="px-4 py-3.5">模型版本</th>
          <th class="px-4 py-3.5">最后通信时间</th>
          <th class="px-4 py-3.5">操作</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for d in devices %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-4 text-slate-400 font-mono">{{ d.Id }}</td>
          <td class="px-4 py-4 text-slate-200 font-medium font-mono">{{ d.MAC }}</td>
          <td class="px-4 py-4 text-slate-300">{{ d.Address or '--' }}</td>
          <td class="px-4 py-4 text-slate-400">{{ d.AreaName or '--' }}</td>
          <td class="px-4 py-4">
            {% if d.status == 'online' %}
            <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-semibold">
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
              在线
            </span>
            {% else %}
            <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-500/10 text-slate-400 border border-slate-550/20 text-xs font-semibold">
              <span class="w-1.5 h-1.5 rounded-full bg-slate-450"></span>
              离线
            </span>
            {% endif %}
          </td>
          <td class="px-4 py-4 text-slate-400 font-mono text-xs">{{ d.ModelInfo or '--' }}</td>
          <td class="px-4 py-4 text-slate-400 font-mono text-xs">{{ d.LastConnectTime or '--' }}</td>
          <td class="px-4 py-4 flex items-center gap-2">
            <button class="bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-400 px-2 py-1 rounded-md font-medium transition active:scale-95 text-xs" onclick="editDevice({{ d.Id }},'{{ d.MAC or '' }}','{{ d.Longitude or '' }}','{{ d.Latitude or '' }}','{{ d.Address or '' }}',{{ d.AreaId or 1 }},'{{ d.ModelInfo or '' }}')">修改</button>
            <button class="bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-400 px-2 py-1 rounded-md font-medium transition active:scale-95 text-xs" onclick="triggerSimulate('device', {{ d.Id }})">模拟故障</button>
            <a href="/admin/device/delete/{{ d.Id }}" class="bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-455 px-2 py-1 rounded-md font-medium transition active:scale-95 text-xs" onclick="return confirm('确认删除?')">删除</a>
          </td>
        </tr>
        {% else %}
        <tr>
          <td colspan="8" class="px-4 py-8 text-center text-slate-500">暂无AI分析盒数据</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- Add Modal -->
<div id="addModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-md rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
        新增AI分析盒
      </h3>
      <button onclick="closeAddModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" action="/admin/device/add" class="p-6 flex flex-col gap-4 text-sm">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">MAC地址</label>
        <input type="text" name="MAC" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm" required>
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">经度</label>
          <input type="text" name="Longitude" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">纬度</label>
          <input type="text" name="Latitude" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">安装位置</label>
        <input type="text" name="Address" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">所属区域</label>
        <select name="AreaId" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
          {% for a in areas %}
          <option value="{{ a.Id }}">{{ a.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">模型信息</label>
        <input type="text" name="ModelInfo" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm" value="YOLOv11-Fire">
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">维护负责人</label>
        <input type="text" name="Maintainer" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeAddModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition text-sm">取消</button>
        <button type="submit" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition text-sm">保存</button>
      </div>
    </form>
  </div>
</div>

<!-- Edit Modal -->
<div id="editModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-md rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
        修改AI分析盒
      </h3>
      <button onclick="closeEditModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" id="editForm" class="p-6 flex flex-col gap-4 text-sm">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">MAC地址</label>
        <input type="text" name="MAC" id="eMAC" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm" required>
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">经度</label>
          <input type="text" name="Longitude" id="eLng" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">纬度</label>
          <input type="text" name="Latitude" id="eLat" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">安装位置</label>
        <input type="text" name="Address" id="eAddr" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">所属区域</label>
        <select name="AreaId" id="eArea" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
          {% for a in areas %}
          <option value="{{ a.Id }}">{{ a.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">模型信息</label>
        <input type="text" name="ModelInfo" id="eModel" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeEditModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition text-sm">取消</button>
        <button type="submit" class="bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-400 px-4 py-2 rounded-lg font-semibold transition text-sm">保存</button>
      </div>
    </form>
  </div>
</div>

<script>
function openAddModal() { document.getElementById('addModal').classList.remove('hidden'); }
function closeAddModal() { document.getElementById('addModal').classList.add('hidden'); }
function closeEditModal() { document.getElementById('editModal').classList.add('hidden'); }
function editDevice(id,mac,lng,lat,addr,area,model){
  document.getElementById('editForm').action='/admin/device/edit/'+id;
  document.getElementById('eMAC').value=mac;
  document.getElementById('eLng').value=lng;
  document.getElementById('eLat').value=lat;
  document.getElementById('eAddr').value=addr;
  document.getElementById('eArea').value=area;
  document.getElementById('eModel').value=model;
  document.getElementById('editModal').classList.remove('hidden');
}
function triggerSimulate(type, id) {
  const codes = type === 'device' 
    ? ['算力异常', '算法崩溃', 'NPU过载'] 
    : ['网络故障', '图像质量差', '视频流中断'];
  const code = prompt(`请输入要模拟的故障类型 (${codes.join('/')}):`, codes[0]);
  if (!code) return;
  const msg = prompt("请输入故障详细描述:", `运维模拟：测试 ${type === 'device' ? 'AI分析盒' : '摄像头'} #${id} 产生 [${code}] 故障告警`);
  if (!msg) return;
  
  fetch('/admin/simulate_error', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({type, id, code, msg})
  })
  .then(r => r.json())
  .then(data => {
    alert(data.msg);
    if(data.code === 200) {
      window.location.reload();
    }
  })
  .catch(err => alert('模拟失败: ' + err));
}
</script>
""", "device")

# 摄像头管理页面模板（维护IP、MAC、RTSP流地址、型号、关联AI分析盒等信息）
CAMERA_TEMPLATE = make_admin_template("摄像头管理", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-3">
      <div class="w-9 h-9 bg-cyan-500/10 border border-cyan-500/20 rounded-xl flex items-center justify-center text-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.15)]">
        📹
      </div>
      摄像头管理
    </h2>
    <button onclick="openAddModal()" class="bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 text-cyan-400 px-4 py-2.5 rounded-lg font-semibold transition active:scale-95 flex items-center gap-1.5 text-xs">
      <span>➕</span> 新增摄像头
    </button>
  </div>

  {% with msgs=get_flashed_messages() %}
  {% if msgs %}
  <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-sm font-semibold">
    {{ msgs[0] }}
  </div>
  {% endif %}
  {% endwith %}

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-sm">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3.5">ID</th>
          <th class="px-4 py-3.5">摄像头名称</th>
          <th class="px-4 py-3.5">IP地址</th>
          <th class="px-4 py-3.5">MAC地址</th>
          <th class="px-4 py-3.5">安装区域</th>
          <th class="px-4 py-3.5">在线状态</th>
          <th class="px-4 py-3.5">设备型号</th>
          <th class="px-4 py-3.5">关联AI盒子</th>
          <th class="px-4 py-3.5">操作</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for c in cameras %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-4 text-slate-400 font-mono">{{ c.Id }}</td>
          <td class="px-4 py-4 text-slate-200 font-medium">{{ c.Name }}</td>
          <td class="px-4 py-4 text-slate-300 font-mono">{{ c.IP or '--' }}</td>
          <td class="px-4 py-4 text-slate-400 font-mono">{{ c.MAC or '--' }}</td>
          <td class="px-4 py-4 text-slate-400">{{ c.AreaName or '--' }}</td>
          <td class="px-4 py-4">
            {% if c.status == 'online' %}
            <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-semibold">
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
              在线
            </span>
            {% else %}
            <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-500/10 text-slate-400 border border-slate-550/20 text-xs font-semibold">
              <span class="w-1.5 h-1.5 rounded-full bg-slate-450"></span>
              离线
            </span>
            {% endif %}
          </td>
          <td class="px-4 py-4 text-slate-400 text-xs">{{ c.Type }}</td>
          <td class="px-4 py-4 text-slate-400 font-mono text-xs">{{ c.DeviceMAC or '--' }}</td>
          <td class="px-4 py-4 flex items-center gap-2">
            <button class="bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-400 px-2 py-1 rounded-md font-medium transition active:scale-95 text-xs" onclick="editCam({{ c.Id }},'{{ c.IP or '' }}','{{ c.MAC or '' }}','{{ c.CameraUrl or '' }}','{{ c.Name or '' }}','{{ c.Longitude or '' }}','{{ c.Latitude or '' }}',{{ c.AreaId or 1 }},'{{ c.Type or '' }}',{{ c.DeviceId or 1 }})">修改</button>
            <button class="bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-400 px-2 py-1 rounded-md font-medium transition active:scale-95 text-xs" onclick="triggerSimulate('camera', {{ c.Id }})">模拟故障</button>
            <a href="/admin/camera/delete/{{ c.Id }}" class="bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-455 px-2 py-1 rounded-md font-medium transition active:scale-95 text-xs" onclick="return confirm('确认删除?')">删除</a>
          </td>
        </tr>
        {% else %}
        <tr>
          <td colspan="9" class="px-4 py-8 text-center text-slate-500">暂无摄像头数据</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- Add Modal -->
<div id="addModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-md rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
        新增摄像头
      </h3>
      <button onclick="closeAddModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" action="/admin/camera/add" class="p-6 flex flex-col gap-4 text-sm">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">摄像头名称</label>
        <input type="text" name="Name" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm" required>
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">IP地址</label>
          <input type="text" name="IP" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">MAC地址</label>
          <input type="text" name="MAC" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">流媒体地址 / URL</label>
        <input type="text" name="CameraUrl" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">经度</label>
          <input type="text" name="Longitude" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">纬度</label>
          <input type="text" name="Latitude" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">区域</label>
        <select name="AreaId" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
          {% for a in areas %}
          <option value="{{ a.Id }}">{{ a.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">型号品牌</label>
        <select name="Type" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
          <option>海康威视</option>
          <option>大华</option>
          <option>宇视</option>
          <option>其他</option>
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">关联AI分析盒</label>
        <select name="DeviceId" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
          {% for d in devices %}
          <option value="{{ d.Id }}">{{ d.MAC }} - {{ d.Address or 'N/A' }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeAddModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition text-sm">取消</button>
        <button type="submit" class="bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 px-4 py-2 rounded-lg font-semibold transition text-sm">保存</button>
      </div>
    </form>
  </div>
</div>

<!-- Edit Modal -->
<div id="editModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
  <div class="glass-panel w-full max-w-md rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col">
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
        修改摄像头
      </h3>
      <button onclick="closeEditModal()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    <form method="post" id="editForm" class="p-6 flex flex-col gap-4 text-sm">
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">摄像头名称</label>
        <input type="text" name="Name" id="eName" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm" required>
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">IP地址</label>
          <input type="text" name="IP" id="eIP" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">MAC地址</label>
          <input type="text" name="MAC" id="eMAC" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">流媒体地址 / URL</label>
        <input type="text" name="CameraUrl" id="eUrl" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">经度</label>
          <input type="text" name="Longitude" id="eLng" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
        <div class="flex flex-col gap-1.5">
          <label class="text-slate-400 font-medium">纬度</label>
          <input type="text" name="Latitude" id="eLat" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
        </div>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">区域</label>
        <select name="AreaId" id="eArea" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
          {% for a in areas %}
          <option value="{{ a.Id }}">{{ a.Name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">型号品牌</label>
        <select name="Type" id="eType" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
          <option>海康威视</option>
          <option>大华</option>
          <option>宇视</option>
          <option>其他</option>
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-slate-400 font-medium">关联AI分析盒</label>
        <select name="DeviceId" id="eDevice" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition text-sm">
          {% for d in devices %}
          <option value="{{ d.Id }}">{{ d.MAC }} - {{ d.Address or 'N/A' }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex justify-end gap-2.5 mt-2">
        <button type="button" onclick="closeEditModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg font-semibold transition text-sm">取消</button>
        <button type="submit" class="bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-400 px-4 py-2 rounded-lg font-semibold transition text-sm">保存</button>
      </div>
    </form>
  </div>
</div>

<script>
function openAddModal() { document.getElementById('addModal').classList.remove('hidden'); }
function closeAddModal() { document.getElementById('addModal').classList.add('hidden'); }
function closeEditModal() { document.getElementById('editModal').classList.add('hidden'); }
function editCam(id,ip,mac,url,name,lng,lat,area,type,did){
  document.getElementById('editForm').action='/admin/camera/edit/'+id;
  document.getElementById('eName').value=name;
  document.getElementById('eIP').value=ip;
  document.getElementById('eMAC').value=mac;
  document.getElementById('eUrl').value=url;
  document.getElementById('eLng').value=lng;
  document.getElementById('eLat').value=lat;
  document.getElementById('eArea').value=area;
  document.getElementById('eType').value=type;
  document.getElementById('eDevice').value=did;
  document.getElementById('editModal').classList.remove('hidden');
}
function triggerSimulate(type, id) {
  const codes = type === 'device' 
    ? ['算力异常', '算法崩溃', 'NPU过载'] 
    : ['网络故障', '图像质量差', '视频流中断'];
  const code = prompt(`请输入要模拟的故障类型 (${codes.join('/')}):`, codes[0]);
  if (!code) return;
  const msg = prompt("请输入故障详细描述:", `运维模拟：测试 ${type === 'device' ? 'AI分析盒' : '摄像头'} #${id} 产生 [${code}] 故障告警`);
  if (!msg) return;
  
  fetch('/admin/simulate_error', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({type, id, code, msg})
  })
  .then(r => r.json())
  .then(data => {
    alert(data.msg);
    if(data.code === 200) {
      window.location.reload();
    }
  })
  .catch(err => alert('模拟失败: ' + err));
}
</script>
""", "camera")

# 报警事件管理页面模板（按时间降序展示，支持按位置/时间/状态筛选，含详情弹窗处理功能）
ALARM_TEMPLATE = make_admin_template("报警事件", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-rose-500 rounded-full shadow-[0_0_8px_#f43f5e]"></span>
      报警事件管理
    </h2>
  </div>

  {% with msgs=get_flashed_messages() %}
  {% if msgs %}
  <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-xs font-semibold">
    {{ msgs[0] }}
  </div>
  {% endif %}
  {% endwith %}

  <!-- Search / Filter Panel -->
  <div class="glass-panel rounded-xl p-4 shadow-md flex flex-wrap gap-4 items-end text-xs border border-slate-800/60">
    <div class="flex flex-col gap-1.5 min-w-[150px] flex-1">
      <label class="text-slate-400 font-medium">发生地点 / 相机</label>
      <input type="text" id="filterLocCam" placeholder="输入地点或相机名称..." class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500/50 transition">
    </div>
    <div class="flex flex-col gap-1.5">
      <label class="text-slate-400 font-medium">开始时间</label>
      <input type="datetime-local" id="filterStart" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
    </div>
    <div class="flex flex-col gap-1.5">
      <label class="text-slate-400 font-medium">结束时间</label>
      <input type="datetime-local" id="filterEnd" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
    </div>
    <div class="flex flex-col gap-1.5 min-w-[120px]">
      <label class="text-slate-400 font-medium">预警状态</label>
      <select id="filterStatus" class="bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
        <option value="">全部</option>
        <option value="1">报警 (未处理)</option>
        <option value="2">待审核</option>
        <option value="3">已审核</option>
      </select>
    </div>
    <div class="flex gap-2 min-w-[150px]">
      <button onclick="applyFilters()" class="flex-1 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 py-1.5 rounded-lg font-semibold transition active:scale-95">筛选</button>
      <button onclick="clearFilters()" class="flex-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 py-1.5 rounded-lg font-semibold transition active:scale-95">重置</button>
    </div>
  </div>

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">现场抓图</th>
          <th class="px-4 py-3">物理位置</th>
          <th class="px-4 py-3">关联摄像头</th>
          <th class="px-4 py-3">报警时间</th>
          <th class="px-4 py-3">报警状态</th>
          <th class="px-4 py-3">处理结果</th>
          <th class="px-4 py-3">处理人</th>
          <th class="px-4 py-3">操作</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for a in alarms %}
        <tr class="hover:bg-slate-900/10 transition duration-150 border-b border-slate-900 cursor-pointer" id="alarm-row-{{ a.Id }}" data-id="{{ a.Id }}" onclick="showAlarmDetail({{ a.Id }})">
          <td class="px-4 py-3 text-slate-400 font-mono">{{ a.Id }}</td>
          <td class="px-4 py-3">
            {% if a.Picture %}
            <div class="w-16 h-10 rounded-md overflow-hidden border border-slate-800 cursor-pointer hover:border-cyan-500/50 transition" onclick="showAlarmDetail({{ a.Id }})">
              <img src="{{ a.Picture }}" class="w-full h-full object-cover">
            </div>
            {% else %}
            <span class="text-slate-600">--</span>
            {% endif %}
          </td>
          <td class="px-4 py-3 text-slate-200 font-medium alarm-location">{{ a.Location or '--' }}</td>
          <td class="px-4 py-3 text-slate-300 alarm-camera">{{ a.CameraName or '--' }}</td>
          <td class="px-4 py-3 text-slate-400 font-mono alarm-time">{{ a.CreatTime }}</td>
          <td class="px-4 py-3 alarm-status" data-status="{{ a.Status }}">
            {% if a.Status=='1' %}
            <span class="px-2.5 py-0.5 rounded-full bg-rose-500/10 text-rose-455 border border-rose-500/20 text-[10px] font-bold">待处理</span>
            {% elif a.OperateResult=='误报无需处理' %}
            <span class="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] font-semibold">排除误报</span>
            {% elif a.OperateResult=='漏报记录' %}
            <span class="px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 text-[10px] font-bold animate-pulse">漏报记录</span>
            {% else %}
            <span class="px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px] font-semibold">已处理</span>
            {% endif %}
          </td>
          <td class="px-4 py-3 text-slate-350 alarm-result font-medium">{{ a.OperateResult or '--' }}</td>
          <td class="px-4 py-3 text-slate-400">{{ a.OperatorName or '--' }}</td>
          <td class="px-4 py-3 flex items-center gap-2">
            <button class="bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 text-cyan-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95" onclick="showAlarmDetail({{ a.Id }})">详情</button>
            {% if a.Status=='1' %}
            <button class="bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95" onclick="showAlarmDetail({{ a.Id }})">处理</button>
            {% endif %}
          </td>
        </tr>
        {% else %}
        <tr>
          <td colspan="9" class="px-4 py-8 text-center text-slate-500">暂无报警事件数据</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- Alarm Detail Modal -->
<div id="detailModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4 animate-fade-in">
  <div class="glass-panel w-full max-w-2xl rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-slate-700/50 overflow-hidden flex flex-col max-h-[90vh] animate-scale-in">
    <!-- Modal Header -->
    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/40">
      <h3 class="text-sm font-bold text-slate-100 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
        预警事件详情
      </h3>
      <button onclick="closeAlarmDetail()" class="text-slate-400 hover:text-slate-200 transition text-lg">&times;</button>
    </div>
    
    <!-- Modal Body -->
    <div class="p-6 overflow-y-auto flex flex-col md:flex-row gap-6 text-xs scrollbar-thin">
      <!-- Left Side: Image & Video -->
      <div class="flex-1 flex flex-col gap-4 animate-fade-in">
        <div class="flex flex-col gap-1.5">
          <span class="text-slate-400 font-semibold">现场图片</span>
          <div class="aspect-video bg-slate-950/50 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center relative">
            <img id="modalAlarmImage" src="" class="w-full h-full object-cover hidden">
            <div id="modalAlarmNoImage" class="text-slate-500 flex flex-col items-center gap-1.5 py-8">
              <svg class="w-8 h-8 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              <span>暂无现场图片</span>
            </div>
          </div>
        </div>
        <div class="flex flex-col gap-1.5">
          <span class="text-slate-400 font-semibold">视频回放</span>
          <div class="aspect-video bg-slate-950/50 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center relative">
            <video id="modalAlarmVideo" src="" class="w-full h-full object-cover hidden" controls autoplay muted></video>
            <div id="modalAlarmNoVideo" class="text-slate-500 flex flex-col items-center gap-1.5 py-8">
              <svg class="w-8 h-8 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
              </svg>
              <span>暂无录像视频</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Right Side: Info & Form -->
      <div class="flex-1 flex flex-col gap-4">
        <!-- Event Metadata -->
        <div class="bg-slate-905/30 border border-slate-800 rounded-xl p-4 flex flex-col gap-2.5">
          <div class="flex justify-between items-center border-b border-slate-800/60 pb-2">
            <span class="text-slate-400">发生地点</span>
            <span class="font-semibold text-slate-200" id="modalAlarmLocation">--</span>
          </div>
          <div class="flex justify-between items-center border-b border-slate-800/60 pb-2">
            <span class="text-slate-400">预警时间</span>
            <span class="font-mono text-slate-300" id="modalAlarmTime">--</span>
          </div>
          <div class="flex justify-between items-center border-b border-slate-800/60 pb-2">
            <span class="text-slate-400">预警类型</span>
            <span id="modalAlarmType" class="font-semibold">--</span>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-slate-400">预警状态</span>
            <span id="modalAlarmStatus" class="font-bold">--</span>
          </div>
        </div>
        
        <!-- Processing Form (For Status = '1') -->
        {% if user.RoleName == '审核人' %}
        <div id="modalProcessForm" class="hidden flex flex-col gap-3 bg-slate-900/20 border border-rose-500/20 rounded-xl p-4 text-center text-rose-400 font-semibold shadow-inner">
          🛡️ 您的角色为审核人，无权处理报警事件。
          <button type="button" onclick="closeAlarmDetail()" class="w-full mt-2 bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-lg font-semibold transition active:scale-95 border border-slate-700">关闭详情</button>
        </div>
        {% else %}
        <form id="modalProcessForm" method="post" action="" class="hidden flex flex-col gap-3">
          <div class="flex flex-col gap-1">
            <label class="text-slate-400 font-medium">紧急程度</label>
            <select name="UrgencyDegree" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
              <option value="普通">普通</option>
              <option value="紧急">紧急</option>
              <option value="特急">特急</option>
            </select>
          </div>
          <div class="flex flex-col gap-1">
            <label class="text-slate-400 font-medium">处理结果</label>
            <select name="OperateResult" class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500/50 transition">
              <option value="火灾已确认并报警">火灾已确认并报警</option>
              <option value="误报无需处理">误报无需处理</option>
              <option value="漏报记录">漏报记录</option>
              <option value="其它已处理">其它已处理</option>
            </select>
          </div>
          <div class="flex flex-col gap-1">
            <label class="text-slate-400 font-medium">处理备注 / 描述</label>
            <textarea name="Description" id="processDescription" rows="2" placeholder="请输入处理备注信息..." class="w-full bg-slate-950/50 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 transition resize-none"></textarea>
          </div>
          <div class="flex gap-2.5 mt-2">
            <button type="submit" class="flex-1 bg-rose-600/90 hover:bg-rose-600 text-white py-2 rounded-lg font-bold transition active:scale-95 shadow-md shadow-rose-950/50">确认处理</button>
            <button type="button" onclick="closeAlarmDetail()" class="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-lg font-semibold transition active:scale-95 border border-slate-700">暂不处理</button>
          </div>
        </form>
        {% endif %}
        
        <!-- Processed Information (For Status = '2' or '3') -->
        <div id="modalProcessedInfo" class="hidden flex flex-col gap-3 bg-slate-900/20 border border-slate-800/40 rounded-xl p-3.5 text-[11px] text-slate-300">
          <div class="flex justify-between border-b border-slate-800/60 pb-1.5">
            <span class="text-slate-400">处理人</span>
            <span class="font-medium text-slate-200" id="modalInfoOperator">--</span>
          </div>
          <div class="flex justify-between border-b border-slate-800/60 pb-1.5">
            <span class="text-slate-400">处理时间</span>
            <span class="font-mono text-slate-200" id="modalInfoTime">--</span>
          </div>
          <div class="flex justify-between border-b border-slate-800/60 pb-1.5">
            <span class="text-slate-400">处理结果</span>
            <span class="font-medium text-orange-400" id="modalInfoResult">--</span>
          </div>
          <div class="flex justify-between border-b border-slate-800/60 pb-1.5">
            <span class="text-slate-400">紧急程度</span>
            <span class="font-semibold text-rose-450" id="modalInfoUrgency">--</span>
          </div>
          <div class="flex flex-col gap-1">
            <span class="text-slate-400">备注描述</span>
            <p class="text-slate-350 leading-relaxed bg-slate-950/40 rounded-lg p-2 border border-slate-850" id="modalInfoDesc">--</p>
          </div>
          
          <!-- Audit Section for Status '2' and authorized users -->
          {% if user.RoleName in ['超级管理员', '审核人'] %}
          <div id="modalAuditActions" class="hidden flex gap-2 mt-2">
            <a id="modalAuditApproveBtn" href="" class="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white text-center py-2 rounded-lg font-semibold transition active:scale-95 shadow-md shadow-emerald-950/30">审核通过</a>
            <a id="modalAuditRejectBtn" href="" class="flex-1 bg-rose-600 hover:bg-rose-500 text-white text-center py-2 rounded-lg font-semibold transition active:scale-95 shadow-md shadow-rose-950/30">驳回</a>
          </div>
          {% endif %}
          
          <button type="button" onclick="closeAlarmDetail()" class="w-full bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-lg font-semibold transition mt-2">关闭</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
// Serialize alarm data to JavaScript object dictionary to avoid escaping problems
const alarmData = {
  {% for a in alarms %}
  "{{ a.Id }}": {
    id: {{ a.Id }},
    time: "{{ a.CreatTime or '' }}",
    location: "{{ (a.Location or a.CameraName or '未知位置')|replace('\\\\', '\\\\\\\\')|replace('"', '\\"') }}",
    picture: "{{ a.Picture or '' }}",
    videoUrl: "{{ a.VideoUrl or '' }}",
    status: "{{ a.Status }}",
    desc: "{{ (a.Description or '')|replace('\\\\', '\\\\\\\\')|replace('"', '\\"')|replace('\r', '')|replace('\n', '\\n') }}",
    urgency: "{{ a.UrgencyDegree if a.UrgencyDegree is not none else '' }}",
    result: "{{ a.OperateResult if a.OperateResult is not none else '' }}",
    operator: "{{ a.OperatorName if a.OperatorName is not none else '' }}",
    operateTime: "{{ a.OperateTime if a.OperateTime is not none else '' }}"
  },
  {% endfor %}
};

function showAlarmDetail(id) {
  try {
    const a = alarmData[id];
    if (!a) return;

    const formEl = document.getElementById('modalProcessForm');
    if (formEl) formEl.action = '/admin/alarm/process/' + id;
    
    const timeEl = document.getElementById('modalAlarmTime');
    if (timeEl) timeEl.textContent = a.time || '--';
    
    const locEl = document.getElementById('modalAlarmLocation');
    if (locEl) locEl.textContent = a.location || '未知位置';
    
    const typeEl = document.getElementById('modalAlarmType');
    if (typeEl) {
      const isSmoke = a.desc && a.desc.includes('烟雾') && !a.desc.includes('火焰');
      if (isSmoke) {
        typeEl.innerHTML = '<span class="text-sky-400 font-bold bg-sky-500/10 border border-sky-500/20 px-1.5 py-0.5 rounded">烟雾报警 (疑似火灾)</span>';
      } else {
        typeEl.innerHTML = '<span class="text-rose-400 font-bold bg-rose-500/10 border border-rose-500/20 px-1.5 py-0.5 rounded">火灾报警</span>';
      }
    }
    
    // Picture
    const img = document.getElementById('modalAlarmImage');
    const noImg = document.getElementById('modalAlarmNoImage');
    if (img && noImg) {
      if (a.picture) {
        img.src = a.picture;
        img.classList.remove('hidden');
        noImg.classList.add('hidden');
      } else {
        img.src = '';
        img.classList.add('hidden');
        noImg.classList.remove('hidden');
      }
    }
    
    // Video
    const video = document.getElementById('modalAlarmVideo');
    const noVideo = document.getElementById('modalAlarmNoVideo');
    if (video && noVideo) {
      if (a.videoUrl) {
        video.src = a.videoUrl;
        video.load();
        video.play().catch(e => console.log("Video play failed:", e));
        video.classList.remove('hidden');
        noVideo.classList.add('hidden');
      } else {
        video.src = '';
        video.classList.add('hidden');
        noVideo.classList.remove('hidden');
      }
    }

    const statusEl = document.getElementById('modalAlarmStatus');
    if (statusEl) {
      if (a.status === '1') {
        statusEl.textContent = '待处理';
        statusEl.className = 'text-rose-455 font-bold';
        if (formEl) formEl.classList.remove('hidden');
        const pInfo = document.getElementById('modalProcessedInfo');
        if (pInfo) pInfo.classList.add('hidden');
      } else {
        let statusText = '已处理';
        let statusClass = 'text-emerald-450 font-bold';
        if (a.result === '误报无需处理') {
          statusText = '排除误报';
          statusClass = 'text-emerald-450 font-bold';
        } else if (a.result === '漏报记录') {
          statusText = '漏报记录';
          statusClass = 'text-amber-450 font-bold animate-pulse';
        } else {
          statusText = '已处理';
          statusClass = 'text-blue-450 font-bold';
        }
        statusEl.textContent = statusText;
        statusEl.className = statusClass;
        
        if (formEl) formEl.classList.add('hidden');
        const pInfo = document.getElementById('modalProcessedInfo');
        if (pInfo) pInfo.classList.remove('hidden');
        
        const opEl = document.getElementById('modalInfoOperator');
        if (opEl) opEl.textContent = a.operator || '系统/未知';
        const opTimeEl = document.getElementById('modalInfoTime');
        if (opTimeEl) opTimeEl.textContent = a.operateTime || '--';
        const opResEl = document.getElementById('modalInfoResult');
        if (opResEl) opResEl.textContent = a.result || '--';
        const opUrgEl = document.getElementById('modalInfoUrgency');
        if (opUrgEl) opUrgEl.textContent = a.urgency || '普通';
        const opDescEl = document.getElementById('modalInfoDesc');
        if (opDescEl) opDescEl.textContent = a.desc || '无备注';
        
        const auditActions = document.getElementById('modalAuditActions');
        if (auditActions) {
          if (a.status === '2') {
            auditActions.classList.remove('hidden');
            const appBtn = document.getElementById('modalAuditApproveBtn');
            if (appBtn) appBtn.href = '/admin/audit/approve/' + id;
            const rejBtn = document.getElementById('modalAuditRejectBtn');
            if (rejBtn) rejBtn.href = '/admin/audit/reject/' + id;
          } else {
            auditActions.classList.add('hidden');
          }
        }
      }
    }

    const modal = document.getElementById('detailModal');
    if (modal) modal.classList.remove('hidden');
  } catch (err) {
    console.error('showAlarmDetail error:', err);
  }
}

function closeAlarmDetail() {
  const modal = document.getElementById('detailModal');
  if (modal) modal.classList.add('hidden');
  const video = document.getElementById('modalAlarmVideo');
  if (video) {
    video.pause();
  }
}

// Client-side filtering logic
function applyFilters() {
  const locCam = document.getElementById('filterLocCam').value.toLowerCase();
  const start = document.getElementById('filterStart').value;
  const end = document.getElementById('filterEnd').value;
  const status = document.getElementById('filterStatus').value;
  
  const startMs = start ? new Date(start).getTime() : 0;
  const endMs = end ? new Date(end).getTime() : 0;
  
  document.querySelectorAll('tbody tr[id^="alarm-row-"]').forEach(row => {
    const id = row.getAttribute('data-id');
    const a = alarmData[id];
    if (!a) return;
    
    let show = true;
    
    if (locCam) {
      const loc = (a.location || '').toLowerCase();
      if (!loc.includes(locCam)) show = false;
    }
    
    if (status) {
      if (a.status !== status) show = false;
    }
    
    if (startMs || endMs) {
      if (a.time) {
        const timeMs = new Date(a.time.replace(' ', 'T')).getTime();
        if (startMs && timeMs < startMs) show = false;
        if (endMs && timeMs > endMs) show = false;
      } else {
        show = false;
      }
    }
    
    if (show) {
      row.classList.remove('hidden');
    } else {
      row.classList.add('hidden');
    }
  });
}

function clearFilters() {
  document.getElementById('filterLocCam').value = '';
  document.getElementById('filterStart').value = '';
  document.getElementById('filterEnd').value = '';
  document.getElementById('filterStatus').value = '';
  
  document.querySelectorAll('tbody tr[id^="alarm-row-"]').forEach(row => {
    row.classList.remove('hidden');
  });
}
</script>
""", "alarm")

# 事件处理审核页面模板（仅审核人和超管可访问，展示待审核事件，支持通过/驳回操作）
AUDIT_TEMPLATE = make_admin_template("事件处理审核", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-emerald-500 rounded-full shadow-[0_0_8px_#10b981]"></span>
      事件处理审核
    </h2>
  </div>

  {% with msgs=get_flashed_messages() %}
  {% if msgs %}
  <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-xs font-semibold">
    {{ msgs[0] }}
  </div>
  {% endif %}
  {% endwith %}

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">物理位置</th>
          <th class="px-4 py-3">关联摄像头</th>
          <th class="px-4 py-3">报警时间</th>
          <th class="px-4 py-3">处理负责人</th>
          <th class="px-4 py-3">处理汇报</th>
          <th class="px-4 py-3">处理时间</th>
          <th class="px-4 py-3">操作选项</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for a in alarms %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ a.Id }}</td>
          <td class="px-4 py-3.5 text-slate-200 font-medium">{{ a.Location or '--' }}</td>
          <td class="px-4 py-3.5 text-slate-300">{{ a.CameraName or '--' }}</td>
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ a.CreatTime }}</td>
          <td class="px-4 py-3.5 text-slate-300 font-medium">{{ a.OperatorName or '--' }}</td>
          <td class="px-4 py-3.5 text-slate-200">{{ a.OperateResult or '--' }}</td>
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ a.OperateTime or '--' }}</td>
          <td class="px-4 py-3.5 flex items-center gap-2">
            <a href="/admin/audit/approve/{{ a.Id }}" class="bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 text-emerald-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95">审核通过</a>
            <a href="/admin/audit/reject/{{ a.Id }}" class="bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-400 px-2.5 py-1 rounded-md font-medium transition active:scale-95">驳回处理</a>
          </td>
        </tr>
        {% else %}
        <tr>
          <td colspan="8" class="px-4 py-8 text-center text-slate-500">暂无待审核事件</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
""", "audit")

# 摄像头故障日志页面模板（展示摄像头异常错误记录，含错误码和详细描述）
CAMERA_ERROR_TEMPLATE = make_admin_template("摄像头故障日志", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-rose-500 rounded-full shadow-[0_0_8px_#f43f5e]"></span>
      摄像头故障日志
    </h2>
  </div>

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">故障摄像头</th>
          <th class="px-4 py-3">MAC地址</th>
          <th class="px-4 py-3">故障发生时间</th>
          <th class="px-4 py-3">故障特征码</th>
          <th class="px-4 py-3">故障详细描述</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for e in errors %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ e.Id }}</td>
          <td class="px-4 py-3.5 text-slate-200 font-medium">{{ e.CameraName or ('监控摄像头 #' ~ e.CameraId) }}</td>
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ e.MAC }}</td>
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ e.CreateTime }}</td>
          <td class="px-4 py-3.5"><span class="px-2 py-0.5 rounded bg-rose-500/10 text-rose-455 border border-rose-500/20 font-mono text-[10px]">{{ e.ErrorCode }}</span></td>
          <td class="px-4 py-3.5 text-slate-300">{{ e.ErrorMsg or '--' }}</td>
        </tr>
        {% else %}
        <tr>
          <td colspan="6" class="px-4 py-8 text-center text-slate-500">暂无摄像头故障日志</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
""", "camera_error")

# AI分析盒故障日志页面模板（展示边缘设备异常错误记录，含错误码和详细描述）
DEVICE_ERROR_TEMPLATE = make_admin_template("AI分析盒故障日志", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-rose-500 rounded-full shadow-[0_0_8px_#f43f5e]"></span>
      AI分析盒故障日志
    </h2>
  </div>

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">故障设备</th>
          <th class="px-4 py-3">MAC地址</th>
          <th class="px-4 py-3">故障发生时间</th>
          <th class="px-4 py-3">故障特征码</th>
          <th class="px-4 py-3">故障详细描述</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for e in errors %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ e.Id }}</td>
          <td class="px-4 py-3.5 text-slate-200 font-medium">{{ e.DeviceAddress or e.DeviceId }}</td>
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ e.DeviceMAC or e.MAC }}</td>
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ e.CreateTime }}</td>
          <td class="px-4 py-3.5"><span class="px-2 py-0.5 rounded bg-rose-500/10 text-rose-455 border border-rose-500/20 font-mono text-[10px]">{{ e.ErrorCode }}</span></td>
          <td class="px-4 py-3.5 text-slate-300">{{ e.ErrorMsg or '--' }}</td>
        </tr>
        {% else %}
        <tr>
          <td colspan="6" class="px-4 py-8 text-center text-slate-500">暂无AI分析盒故障日志</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
""", "device_error")

# 安全访问日志页面模板（展示用户登录记录，包含登录时间、IP、方式等信息）
ACCESS_LOG_TEMPLATE = make_admin_template("访问安全日志", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-cyan-500 rounded-full shadow-[0_0_8px_#06b6d4]"></span>
      安全访问日志
    </h2>
  </div>

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">登录用户</th>
          <th class="px-4 py-3">登录时间</th>
          <th class="px-4 py-3">客户端IP</th>
          <th class="px-4 py-3">访问途径 / 登录方式</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for l in logs %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ l.Id }}</td>
          <td class="px-4 py-3.5 text-slate-200 font-medium">{{ l.UserName or l.UserId }}</td>
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ l.LoginTime }}</td>
          <td class="px-4 py-3.5 text-slate-300 font-mono">{{ l.LoginInIp }}</td>
          <td class="px-4 py-3.5 text-slate-400">{{ l.LoginType }}</td>
        </tr>
        {% else %}
        <tr>
          <td colspan="5" class="px-4 py-8 text-center text-slate-500">暂无访问日志</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
""", "access_log")

# 业务操作日志页面模板（展示用户的增删改操作记录，包含功能模块、操作类型、变更内容和执行人信息）
OPERATE_LOG_TEMPLATE = make_admin_template("操作日志", """
<div class="flex flex-col gap-6">
  <div class="flex justify-between items-center border-b border-slate-800 pb-4">
    <h2 class="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
      <span class="w-1.5 h-4.5 bg-cyan-500 rounded-full shadow-[0_0_8px_#06b6d4]"></span>
      业务操作日志
    </h2>
  </div>

  <!-- Table -->
  <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/30 shadow-xl">
    <table class="w-full text-left border-collapse text-xs">
      <thead>
        <tr class="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-semibold tracking-wider">
          <th class="px-4 py-3">ID</th>
          <th class="px-4 py-3">功能模块</th>
          <th class="px-4 py-3">操作类型</th>
          <th class="px-4 py-3">变更细节 / 内容</th>
          <th class="px-4 py-3">执行时间</th>
          <th class="px-4 py-3">操作执行人</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-900">
        {% for l in logs %}
        <tr class="hover:bg-slate-900/20 transition duration-150">
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ l.Id }}</td>
          <td class="px-4 py-3.5 text-slate-200 font-medium">{{ l.MenuName }}</td>
          <td class="px-4 py-3.5"><span class="px-2 py-0.5 rounded text-[10px] font-semibold {% if l.Type == '删除' %}bg-rose-500/10 text-rose-400 border border-rose-500/20{% elif l.Type == '修改' %}bg-amber-500/10 text-amber-400 border border-amber-500/20{% else %}bg-emerald-500/10 text-emerald-400 border border-emerald-500/20{% endif %}">{{ l.Type }}</span></td>
          <td class="px-4 py-3.5 text-slate-300 font-mono max-w-[300px] truncate" title="{{ l.ContentNew }}">{{ l.ContentNew }}</td>
          <td class="px-4 py-3.5 text-slate-400 font-mono">{{ l.CreateTime }}</td>
          <td class="px-4 py-3.5 text-slate-200">{{ l.UserName or l.UserId }}</td>
        </tr>
        {% else %}
        <tr>
          <td colspan="6" class="px-4 py-8 text-center text-slate-550">暂无操作日志</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
""", "operate_log")





# --- 应用入口：初始化数据库并启动 Flask Web 服务器 ---
if __name__ == "__main__":
    init_db()  # 创建表结构并填充种子数据（仅在首次运行时）
    logger.info("Starting Web Management Server...")
    logger.info("访问地址: http://0.0.0.0:5000")
    logger.info("管理员: admin / 123456")
    logger.info("处理人: chuli001 / 123456")
    app.run(host="0.0.0.0", port=5000, debug=True)
