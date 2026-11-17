"""SQLite persistence shared by the API and dashboard. All patients here are SYNTHETIC."""
import sqlite3, uuid, os, json, datetime as dt
def _db(): return os.environ.get("MEDSCOPE_DB", "medscope.db")
def conn():
    c = sqlite3.connect(_db()); c.row_factory = sqlite3.Row; return c
def now(): return dt.datetime.now().isoformat(timespec="seconds")
def init():
    with conn() as c:
        c.executescript("""
create table if not exists patients(pid text primary key,age int,sex text,history text,synthetic int default 1,created text);
create table if not exists cases(id integer primary key autoincrement,pid text,summary text,priority text,score int,status text default 'Pending review',created text);
create table if not exists audit(id integer primary key autoincrement,ts text,actor text,action text,detail text);""")
def rows(sql, a=()):
    with conn() as c: return [dict(r) for r in c.execute(sql, a).fetchall()]
def log(actor, action, detail=""):
    with conn() as c: c.execute("insert into audit(ts,actor,action,detail) values(?,?,?,?)", (now(), actor, action, detail))
def add_patient(age, sex, history=""):
    pid = "SYN-" + uuid.uuid4().hex[:8].upper()
    with conn() as c: c.execute("insert into patients values(?,?,?,?,1,?)", (pid, age, sex, history, now()))
    log("system", "patient_registered", pid); return pid
def add_case(pid, summary, priority, score):
    with conn() as c:
        cur = c.execute("insert into cases(pid,summary,priority,score,created) values(?,?,?,?,?)", (pid, json.dumps(summary), priority, score, now()))
    log("system", "case_created", f"case {cur.lastrowid} {pid} {priority}"); return cur.lastrowid
def set_status(case_id, status, actor="clinician"):
    with conn() as c: c.execute("update cases set status=? where id=?", (status, case_id))
    log(actor, "clinician_decision", f"case {case_id} -> {status}")
