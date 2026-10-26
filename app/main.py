from __future__ import annotations

import csv
import hashlib
import io
import os
import secrets
import contextvars
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sqlalchemy import (Date, DateTime, Float, ForeignKey, Integer, String, create_engine,
                        select, func)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
load_dotenv(PROJECT / ".env")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or f"sqlite:///{PROJECT / 'medisync.db'}"
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
Session = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Hospital(Base):
    __tablename__ = "hospitals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    city: Mapped[str] = mapped_column(String(80))


class Medicine(Base):
    __tablename__ = "medicines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    category: Mapped[str] = mapped_column(String(80), default="General")
    unit: Mapped[str] = mapped_column(String(30), default="units")


class Inventory(Base):
    __tablename__ = "inventory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hospital_id: Mapped[int] = mapped_column(ForeignKey("hospitals.id"))
    medicine_id: Mapped[int] = mapped_column(ForeignKey("medicines.id"))
    batch_code: Mapped[str] = mapped_column(String(50))
    quantity: Mapped[int] = mapped_column(Integer)
    reorder_level: Mapped[int] = mapped_column(Integer, default=50)
    expiry_date: Mapped[date] = mapped_column(Date)


class Consumption(Base):
    __tablename__ = "consumption"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hospital_id: Mapped[int] = mapped_column(ForeignKey("hospitals.id"))
    medicine_id: Mapped[int] = mapped_column(ForeignKey("medicines.id"))
    recorded_on: Mapped[date] = mapped_column(Date)
    quantity: Mapped[int] = mapped_column(Integer)


class Audit(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    action: Mapped[str] = mapped_column(String(80))
    detail: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StockMovement(Base):
    __tablename__ = "stock_movements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inventory_id: Mapped[int] = mapped_column(ForeignKey("inventory.id"))
    movement_type: Mapped[str] = mapped_column(String(30))
    quantity: Mapped[int] = mapped_column(Integer)
    quantity_before: Mapped[int] = mapped_column(Integer)
    quantity_after: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Transfer(Base):
    __tablename__ = "transfers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    medicine_id: Mapped[int] = mapped_column(ForeignKey("medicines.id"))
    from_hospital_id: Mapped[int] = mapped_column(ForeignKey("hospitals.id"))
    to_hospital_id: Mapped[int] = mapped_column(ForeignKey("hospitals.id"))
    donor_inventory_id: Mapped[int] = mapped_column(ForeignKey("inventory.id"))
    quantity: Mapped[int] = mapped_column(Integer)
    batch_code: Mapped[str] = mapped_column(String(50))
    expiry_date: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), default="Requested")
    decision_note: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True)
    display_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(40))
    password_hash: Mapped[str] = mapped_column(String(256))
    active: Mapped[int] = mapped_column(Integer, default=1)


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=1, max_length=200)


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    display_name: str = Field(min_length=2, max_length=120)
    role: Literal["Admin", "Hospital Manager", "Pharmacist", "Auditor"]
    password: str = Field(min_length=8, max_length=200)


ROLE_PERMISSIONS = {
    "Admin": {"read", "inventory", "movement", "transfer_request", "transfer_review", "transfer_complete", "import", "simulate", "users"},
    "Hospital Manager": {"read", "inventory", "movement", "transfer_request", "transfer_review", "transfer_complete", "import", "simulate"},
    "Pharmacist": {"read", "movement", "transfer_request", "simulate"},
    "Auditor": {"read"},
}
TOKENS: dict[str, int] = {}
active_actor = contextvars.ContextVar("medisync_actor", default="System")


def password_digest(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240_000)
    return salt.hex() + "." + derived.hex()


def password_matches(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split(".", 1)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 240_000).hex()
        return secrets.compare_digest(actual, digest_hex)
    except (ValueError, TypeError):
        return False


class SimulationRequest(BaseModel):
    medicine_id: int
    hospital_id: int
    demand_change_pct: float = Field(default=0, ge=-90, le=300)
    supply_change_pct: float = Field(default=0, ge=-90, le=300)
    horizon_days: int = Field(default=30, ge=7, le=90)


class StockMovementCreate(BaseModel):
    movement_type: Literal["received", "issued", "set"]
    quantity: int = Field(ge=0)
    reason: str = Field(default="", max_length=300)


class TransferCreate(BaseModel):
    donor_inventory_id: int = Field(ge=1)
    to_hospital_id: int = Field(ge=1)
    quantity: int = Field(gt=0)

class InventoryCreate(BaseModel):
    hospital: str = Field(min_length=2, max_length=120)
    city: str = Field(min_length=2, max_length=80)
    medicine: str = Field(min_length=2, max_length=120)
    category: str = Field(default="General", max_length=80)
    unit: str = Field(default="packs", max_length=30)
    batch_code: str | None = Field(default=None, max_length=50)
    quantity: int = Field(ge=0)
    reorder_level: int = Field(ge=0)
    expiry_date: date


def audit(db, action: str, detail: str):
    db.add(Audit(action=action, detail=f"[{active_actor.get()}] {detail}"[:500]))
    db.commit()


def seed_demo(db):
    if db.scalar(select(func.count(Hospital.id))):
        return
    hospitals = [Hospital(name="Central City Hospital", city="Bengaluru"),
                 Hospital(name="Riverside Medical Centre", city="Mysuru"),
                 Hospital(name="Northside General", city="Hubballi")]
    meds = [Medicine(name=n, category=c, unit="packs") for n, c in [
        ("Amoxicillin 500mg", "Antibiotic"), ("Paracetamol 500mg", "Analgesic"),
        ("Insulin Glargine", "Diabetes"), ("ORS Sachets", "Essential care"),
        ("Azithromycin 250mg", "Antibiotic"), ("Salbutamol Inhaler", "Respiratory")]]
    db.add_all(hospitals + meds); db.flush()
    today = date.today()
    stocks = [[12, 180, 28, 310, 55, 36], [260, 75, 88, 90, 8, 120], [110, 220, 42, 180, 160, 48]]
    rng = np.random.default_rng(19)
    for hi, hospital in enumerate(hospitals):
        for mi, medicine in enumerate(meds):
            qty = int(stocks[hi][mi])
            days_exp = [110, 220, 40, 300, 15, 180][(mi + hi) % 6]
            db.add(Inventory(hospital_id=hospital.id, medicine_id=medicine.id,
                             batch_code=f"DEMO-{hi+1}{mi+1}-A", quantity=qty,
                             reorder_level=[80, 90, 55, 100, 70, 40][mi],
                             expiry_date=today + timedelta(days=days_exp)))
            base = max(2, qty // 28)
            for day in range(120):
                seasonal = 1.0 + 0.18 * np.sin(day / 11 + mi)
                trend = 1 + day * (0.0007 if mi in (0, 4) else 0.0002)
                amount = max(0, int(rng.normal(base * seasonal * trend, max(1, base * .14))))
                db.add(Consumption(hospital_id=hospital.id, medicine_id=medicine.id,
                                   recorded_on=today - timedelta(days=119-day), quantity=amount))
    db.commit()
    audit(db, "DEMO_SEEDED", "Synthetic sample records loaded; replace with real validated data before operational use.")


app = FastAPI(title="MediSync AI", version="0.1.0", description="Medicine inventory planning MVP; demo data is synthetic.")
Base.metadata.create_all(engine)
with Session() as session:
    seed_demo(session)
    if not session.scalar(select(func.count(User.id))):
        demo_users = [
            ("admin", "MediSync Administrator", "Admin", "Admin@123"),
            ("manager", "Hospital Manager", "Hospital Manager", "Manager@123"),
            ("pharmacist", "Hospital Pharmacist", "Pharmacist", "Pharma@123"),
            ("auditor", "Read-only Auditor", "Auditor", "Audit@123"),
        ]
        session.add_all([User(username=u, display_name=n, role=r, password_hash=password_digest(p))
                         for u, n, r, p in demo_users])
        session.commit()


@app.middleware("http")
async def require_role_access(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/") or path in {"/api/health", "/api/auth/login"}:
        return await call_next(request)
    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    user_id = TOKENS.get(token)
    if not user_id:
        from starlette.responses import JSONResponse
        return JSONResponse({"detail": "Please sign in to use MediSync."}, status_code=401)
    with Session() as db:
        user = db.get(User, user_id)
        if not user or not user.active:
            TOKENS.pop(token, None)
            from starlette.responses import JSONResponse
            return JSONResponse({"detail": "This account is inactive. Contact an administrator."}, status_code=401)
        role, username = user.role, user.username
    permission = "read"
    if path == "/api/users":
        permission = "users"
    elif path == "/api/auth/logout":
        permission = "read"
    elif request.method not in {"GET", "HEAD", "OPTIONS"}:
        if path == "/api/inventory" and request.method == "POST": permission = "inventory"
        elif path.endswith("/movement") and request.method == "POST": permission = "movement"
        elif path == "/api/import/csv" and request.method == "POST": permission = "import"
        elif path == "/api/simulate" and request.method == "POST": permission = "simulate"
        elif path == "/api/transfers" and request.method == "POST": permission = "transfer_request"
        elif path.startswith("/api/transfers/") and path.endswith("/complete"): permission = "transfer_complete"
        elif path.startswith("/api/transfers/") and request.method == "POST": permission = "transfer_review"
        elif path.startswith("/api/users"): permission = "users"
        else: permission = "inventory"
    if permission not in ROLE_PERMISSIONS.get(role, set()):
        from starlette.responses import JSONResponse
        return JSONResponse({"detail": f"{role} role does not have permission for this action."}, status_code=403)
    ctx = active_actor.set(f"{username}/{role}")
    try:
        return await call_next(request)
    finally:
        active_actor.reset(ctx)


def rows_for(db):
    stmt = select(Inventory, Hospital, Medicine).join(Hospital).join(Medicine)
    return db.execute(stmt).all()


def forecast(db, hospital_id: int, medicine_id: int, horizon: int = 30):
    history = db.execute(select(Consumption.recorded_on, Consumption.quantity)
                          .where(Consumption.hospital_id == hospital_id,
                                 Consumption.medicine_id == medicine_id)
                          .order_by(Consumption.recorded_on)).all()
    if len(history) >= 14:
        frame = pd.DataFrame(history, columns=["day", "quantity"])
        frame["x"] = np.arange(len(frame))
        model = LinearRegression().fit(frame[["x"]], frame["quantity"])
        future = model.predict(pd.DataFrame({"x": np.arange(len(frame), len(frame) + horizon)}))
        method = "Linear trend baseline (scikit-learn)"
        # Dampen extrapolation and floor at zero; simple baseline, not validated forecasting.
        mean = max(0.0, float(np.mean(np.maximum(future, 0))))
    else:
        mean = float(np.mean([q for _, q in history])) if history else 0.0
        method = "Historical average fallback"
    return round(mean * horizon), round(mean, 1), method


def inventory_payload(db):
    today = date.today()
    result = []
    for inv, hospital, medicine in rows_for(db):
        demand_total, daily, method = forecast(db, inv.hospital_id, inv.medicine_id)
        days_left = (inv.expiry_date - today).days
        cover = round(inv.quantity / daily, 1) if daily else 999
        projected = max(0, demand_total - inv.quantity)
        risk = "High" if (inv.quantity <= 0 or cover < 7) else "Medium" if (cover < 21 or inv.quantity <= inv.reorder_level) else "Low"
        result.append({"inventory_id": inv.id, "hospital_id": hospital.id, "hospital": hospital.name,
                       "city": hospital.city, "medicine_id": medicine.id, "medicine": medicine.name,
                       "category": medicine.category, "unit": medicine.unit, "batch_code": inv.batch_code,
                       "quantity": inv.quantity, "reorder_level": inv.reorder_level,
                       "expiry_date": inv.expiry_date.isoformat(), "days_to_expiry": days_left,
                       "days_of_cover": cover, "forecast_30d": demand_total, "daily_forecast": daily,
                       "shortage_risk": risk, "reorder_suggestion": max(0, int(demand_total * 1.2 - inv.quantity)),
                       "model": method})
    return result


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "database": "sqlite" if DATABASE_URL.startswith("sqlite") else "mysql",
            "synthetic_demo_data": True, "version": app.version}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    with Session() as db:
        user = db.scalar(select(User).where(func.lower(User.username) == req.username.strip().lower()))
        if not user or not user.active or not password_matches(req.password, user.password_hash):
            raise HTTPException(401, "Username or password is incorrect")
        token = secrets.token_urlsafe(32)
        TOKENS[token] = user.id
        return {"access_token": token, "user": {"username": user.username, "display_name": user.display_name, "role": user.role}}


@app.get("/api/auth/me")
def auth_me(request: Request):
    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    with Session() as db:
        user = db.get(User, TOKENS.get(token)) if token else None
        if not user or not user.active: raise HTTPException(401, "Please sign in again")
        return {"username": user.username, "display_name": user.display_name, "role": user.role}


@app.post("/api/auth/logout")
def logout(request: Request):
    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    TOKENS.pop(token, None)
    return {"ok": True}


@app.get("/api/users")
def list_users():
    with Session() as db:
        return [{"id": u.id, "username": u.username, "display_name": u.display_name, "role": u.role,
                 "active": bool(u.active)} for u in db.scalars(select(User).order_by(User.username)).all()]


@app.post("/api/users", status_code=201)
def create_user(req: UserCreate):
    with Session() as db:
        if db.scalar(select(User).where(func.lower(User.username) == req.username.lower())):
            raise HTTPException(409, "That username is already in use")
        user = User(username=req.username.strip(), display_name=req.display_name.strip(),
                    role=req.role, password_hash=password_digest(req.password))
        db.add(user); db.flush()
        audit(db, "USER_CREATED", f"Created {req.role} account {user.username}")
        return {"id": user.id, "username": user.username, "display_name": user.display_name,
                "role": user.role, "active": True}


@app.get("/api/dashboard")
def dashboard():
    with Session() as db:
        items = inventory_payload(db)
        transfers = recommendations(items)
        return {"kpis": {"hospitals": len({r['hospital_id'] for r in items}),
                          "medicines": len({r['medicine_id'] for r in items}),
                          "high_risk": sum(r['shortage_risk'] == "High" for r in items),
                          "expiring_30d": sum(r['days_to_expiry'] <= 30 for r in items),
                          "units_on_hand": sum(r['quantity'] for r in items)},
                "inventory": items, "redistribution": transfers,
                "audit": audit_payload(db), "demo_data": True}


def recommendations(items):
    recs = []
    for need in items:
        gap = max(0, need["forecast_30d"] - need["quantity"])
        if gap <= 0: continue
        donors = [x for x in items if x["medicine_id"] == need["medicine_id"] and
                  x["hospital_id"] != need["hospital_id"] and x["days_to_expiry"] >= 0 and
                  x["quantity"] > max(x["reorder_level"], x["forecast_30d"])]
        for donor in sorted(donors, key=lambda x: x["days_to_expiry"]):
            safe_surplus = max(0, donor["quantity"] - max(donor["reorder_level"], donor["forecast_30d"]))
            move = min(gap, safe_surplus)
            if move:
                recs.append({"medicine": need["medicine"], "medicine_id": need["medicine_id"],
                             "to_hospital_id": need["hospital_id"], "donor_inventory_id": donor["inventory_id"],
                             "from_hospital": donor["hospital"], "to_hospital": need["hospital"],
                             "suggested_quantity": int(move), "donor_stock_after": donor["quantity"] - int(move),
                             "expires": donor["expiry_date"], "reason": "Donor remains above forecast and reorder buffer; FEFO batch selected."})
                gap -= move
            if gap <= 0: break
    return recs


def audit_payload(db):
    return [{"id": a.id, "action": a.action, "detail": a.detail, "created_at": a.created_at.isoformat()}
            for a in db.scalars(select(Audit).order_by(Audit.created_at.desc()).limit(12)).all()]


@app.get("/api/inventory")
def get_inventory():
    with Session() as db: return inventory_payload(db)


@app.post("/api/inventory/{inventory_id}/movement")
def record_stock_movement(inventory_id: int, req: StockMovementCreate):
    with Session() as db:
        batch = db.get(Inventory, inventory_id)
        if not batch: raise HTTPException(404, "Inventory batch not found")
        before = batch.quantity
        if req.movement_type in ("received", "issued") and batch.expiry_date < date.today():
            raise HTTPException(409, "Expired batches cannot receive or issue stock")
        if req.movement_type == "received":
            if req.quantity < 1: raise HTTPException(400, "Received quantity must be greater than zero")
            after = before + req.quantity
        elif req.movement_type == "issued":
            if req.quantity < 1: raise HTTPException(400, "Issued quantity must be greater than zero")
            if req.quantity > before: raise HTTPException(409, "Issue quantity exceeds batch stock")
            after = before - req.quantity
        else:
            after = req.quantity
            if after == before: raise HTTPException(400, "New quantity is the same as current stock")
        delta = after - before
        batch.quantity = after
        movement = StockMovement(inventory_id=batch.id, movement_type=req.movement_type,
                                 quantity=abs(delta), quantity_before=before,
                                 quantity_after=after, reason=req.reason.strip())
        db.add(movement); db.flush()
        hospital = db.get(Hospital, batch.hospital_id)
        medicine = db.get(Medicine, batch.medicine_id)
        audit(db, "STOCK_" + req.movement_type.upper(),
              f"{medicine.name} at {hospital.name}, batch {batch.batch_code}: {before} → {after}. {req.reason.strip()}")
        return {"movement_id": movement.id, "inventory_id": batch.id, "quantity_before": before,
                "quantity_after": after, "movement_type": req.movement_type}


@app.get("/api/stock-movements")
def get_stock_movements():
    with Session() as db:
        movements = db.scalars(select(StockMovement).order_by(StockMovement.created_at.desc()).limit(50)).all()
        result = []
        for m in movements:
            batch = db.get(Inventory, m.inventory_id)
            hospital = db.get(Hospital, batch.hospital_id) if batch else None
            medicine = db.get(Medicine, batch.medicine_id) if batch else None
            result.append({"id": m.id, "movement_type": m.movement_type, "quantity": m.quantity,
                           "quantity_before": m.quantity_before, "quantity_after": m.quantity_after,
                           "reason": m.reason, "created_at": m.created_at.isoformat(),
                           "hospital": hospital.name if hospital else "", "medicine": medicine.name if medicine else "",
                           "batch_code": batch.batch_code if batch else ""})
        return result


def transfer_payload(db):
    result = []
    transfers = db.scalars(select(Transfer).order_by(Transfer.created_at.desc()).limit(100)).all()
    for t in transfers:
        medicine = db.get(Medicine, t.medicine_id)
        donor = db.get(Hospital, t.from_hospital_id)
        receiver = db.get(Hospital, t.to_hospital_id)
        result.append({"id": t.id, "medicine": medicine.name if medicine else "",
                       "medicine_id": t.medicine_id, "from_hospital": donor.name if donor else "",
                       "from_hospital_id": t.from_hospital_id, "to_hospital": receiver.name if receiver else "",
                       "to_hospital_id": t.to_hospital_id, "donor_inventory_id": t.donor_inventory_id,
                       "quantity": t.quantity, "batch_code": t.batch_code,
                       "expiry_date": t.expiry_date.isoformat(), "status": t.status,
                       "decision_note": t.decision_note, "created_at": t.created_at.isoformat()})
    return result


@app.get("/api/transfers")
def get_transfers():
    with Session() as db: return transfer_payload(db)


@app.post("/api/transfers", status_code=201)
def request_transfer(req: TransferCreate):
    with Session() as db:
        donor = db.get(Inventory, req.donor_inventory_id)
        if not donor: raise HTTPException(404, "Donor inventory batch not found")
        if donor.expiry_date < date.today(): raise HTTPException(409, "Expired batches cannot be transferred")
        if donor.hospital_id == req.to_hospital_id: raise HTTPException(400, "Choose a different receiving hospital")
        receiver = db.get(Hospital, req.to_hospital_id)
        if not receiver: raise HTTPException(404, "Receiving hospital not found")
        items = inventory_payload(db)
        donor_view = next((x for x in items if x["inventory_id"] == donor.id), None)
        if not donor_view: raise HTTPException(404, "Donor stock could not be evaluated")
        pending_out = db.scalar(select(func.coalesce(func.sum(Transfer.quantity), 0)).where(Transfer.donor_inventory_id == donor.id, Transfer.status.in_(["Requested", "Approved"]))) or 0
        surplus = max(0, donor.quantity - max(donor_view["reorder_level"], donor_view["forecast_30d"]) - pending_out)
        target_rows = [x for x in items if x["hospital_id"] == receiver.id and x["medicine_id"] == donor.medicine_id]
        target_demand = max((x["forecast_30d"] for x in target_rows), default=0)
        target_stock = sum(x["quantity"] for x in target_rows)
        pending_in = db.scalar(select(func.coalesce(func.sum(Transfer.quantity), 0)).where(Transfer.to_hospital_id == receiver.id, Transfer.medicine_id == donor.medicine_id, Transfer.status.in_(["Requested", "Approved"]))) or 0
        target_gap = max(0, target_demand - target_stock - pending_in)
        if req.quantity > surplus: raise HTTPException(409, f"Requested amount exceeds safe donor surplus ({surplus})")
        if req.quantity > target_gap: raise HTTPException(409, f"Requested amount exceeds estimated receiving need ({target_gap})")
        rec = Transfer(medicine_id=donor.medicine_id, from_hospital_id=donor.hospital_id,
                       to_hospital_id=receiver.id, donor_inventory_id=donor.id, quantity=req.quantity,
                       batch_code=donor.batch_code, expiry_date=donor.expiry_date,
                       status="Requested", decision_note="Awaiting local review")
        db.add(rec); db.flush()
        medicine = db.get(Medicine, donor.medicine_id)
        audit(db, "TRANSFER_REQUESTED", f"{req.quantity} {medicine.name}: {donor_view['hospital']} → {receiver.name}; batch {donor.batch_code}")
        return transfer_payload(db)[0]


def decide_transfer(transfer_id: int, status: str, note: str):
    with Session() as db:
        t = db.get(Transfer, transfer_id)
        if not t: raise HTTPException(404, "Transfer request not found")
        if t.status != "Requested": raise HTTPException(409, "Only requested transfers can be approved or rejected")
        t.status = status; t.decision_note = note
        audit(db, "TRANSFER_" + status.upper(), f"Transfer #{t.id}: {note or status}")
        return next(x for x in transfer_payload(db) if x["id"] == t.id)


@app.post("/api/transfers/{transfer_id}/approve")
def approve_transfer(transfer_id: int):
    return decide_transfer(transfer_id, "Approved", "Approved for stock movement")


@app.post("/api/transfers/{transfer_id}/reject")
def reject_transfer(transfer_id: int):
    return decide_transfer(transfer_id, "Rejected", "Rejected after review")


@app.post("/api/transfers/{transfer_id}/complete")
def complete_transfer(transfer_id: int):
    with Session() as db:
        t = db.get(Transfer, transfer_id)
        if not t: raise HTTPException(404, "Transfer request not found")
        if t.status != "Approved": raise HTTPException(409, "Approve this transfer before completing it")
        donor = db.get(Inventory, t.donor_inventory_id)
        if not donor or donor.quantity < t.quantity: raise HTTPException(409, "Donor batch no longer has enough stock")
        if t.expiry_date < date.today(): raise HTTPException(409, "This batch expired before transfer completion")
        items = inventory_payload(db)
        view = next((x for x in items if x["inventory_id"] == donor.id), None)
        if not view or donor.quantity - t.quantity < max(view["reorder_level"], view["forecast_30d"]):
            raise HTTPException(409, "Donor would fall below its safety buffer; reject and request a smaller transfer")
        target_rows = [x for x in items if x["hospital_id"] == t.to_hospital_id and x["medicine_id"] == t.medicine_id]
        target_demand = max((x["forecast_30d"] for x in target_rows), default=0)
        target_stock = sum(x["quantity"] for x in target_rows)
        pending_out = db.scalar(select(func.coalesce(func.sum(Transfer.quantity), 0)).where(Transfer.donor_inventory_id == donor.id, Transfer.status.in_(["Requested", "Approved"]), Transfer.id != t.id)) or 0
        pending_in = db.scalar(select(func.coalesce(func.sum(Transfer.quantity), 0)).where(Transfer.to_hospital_id == t.to_hospital_id, Transfer.medicine_id == t.medicine_id, Transfer.status.in_(["Requested", "Approved"]), Transfer.id != t.id)) or 0
        if donor.quantity - t.quantity - pending_out < max(view["reorder_level"], view["forecast_30d"]):
            raise HTTPException(409, "Other pending transfers use the donor safety buffer; reject and request a smaller transfer")
        if target_stock + t.quantity + pending_in > target_demand:
            raise HTTPException(409, "Receiving hospital no longer needs this full amount; reject and request a smaller transfer")
        before = donor.quantity; donor.quantity -= t.quantity
        destination_code = f"TRF-{t.id}-{t.batch_code}"[:50]
        received = Inventory(hospital_id=t.to_hospital_id, medicine_id=t.medicine_id,
                             batch_code=destination_code, quantity=t.quantity,
                             reorder_level=0, expiry_date=t.expiry_date)
        db.add(received); db.flush()
        db.add_all([StockMovement(inventory_id=donor.id, movement_type="transfer_out", quantity=t.quantity,
                                  quantity_before=before, quantity_after=donor.quantity, reason=f"Transfer #{t.id}"),
                    StockMovement(inventory_id=received.id, movement_type="transfer_in", quantity=t.quantity,
                                  quantity_before=0, quantity_after=t.quantity, reason=f"Transfer #{t.id}; source batch {t.batch_code}")])
        t.status = "Completed"; t.decision_note = f"Completed; receiver batch {destination_code}"
        audit(db, "TRANSFER_COMPLETED", f"Transfer #{t.id}: {t.quantity} {db.get(Medicine, t.medicine_id).name} from {db.get(Hospital, t.from_hospital_id).name} to {db.get(Hospital, t.to_hospital_id).name}")
        return next(x for x in transfer_payload(db) if x["id"] == t.id)


@app.post("/api/inventory", status_code=201)
def add_inventory(req: InventoryCreate):
    with Session() as db:
        hospital = db.scalar(select(Hospital).where(func.lower(Hospital.name) == req.hospital.lower()))
        if hospital and hospital.city.lower() != req.city.lower():
            raise HTTPException(409, f"{hospital.name} is already registered in {hospital.city}")
        if not hospital:
            hospital = Hospital(name=req.hospital.strip(), city=req.city.strip())
            db.add(hospital); db.flush()
        medicine = db.scalar(select(Medicine).where(func.lower(Medicine.name) == req.medicine.lower()))
        if not medicine:
            medicine = Medicine(name=req.medicine.strip(), category=req.category.strip(), unit=req.unit.strip())
            db.add(medicine); db.flush()
        batch_code = (req.batch_code or "").strip() or f"MAN-{datetime.utcnow():%y%m%d%H%M%S}"
        batch = Inventory(hospital_id=hospital.id, medicine_id=medicine.id, batch_code=batch_code,
                          quantity=req.quantity, reorder_level=req.reorder_level, expiry_date=req.expiry_date)
        db.add(batch); db.flush()
        audit(db, "STOCK_ADDED", f"Added {req.quantity} {medicine.name} at {hospital.name} in batch {batch_code}")
        return {"inventory_id": batch.id, "hospital": hospital.name, "city": hospital.city,
                "medicine": medicine.name, "batch_code": batch_code, "quantity": batch.quantity,
                "expiry_date": batch.expiry_date.isoformat()}


@app.get("/api/redistribution")
def get_redistribution():
    with Session() as db: return recommendations(inventory_payload(db))


@app.get("/api/audit")
def get_audit():
    with Session() as db: return audit_payload(db)


@app.post("/api/import/csv")
def import_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file")
    try:
        raw_csv = file.file.read()
        if len(raw_csv) > 5_000_000:
            raise ValueError("CSV file is too large (maximum 5 MB)")
        df = pd.read_csv(io.BytesIO(raw_csv))
        needed = {"hospital", "city", "medicine", "category", "quantity", "reorder_level", "expiry_date"}
        if not needed.issubset(set(df.columns.str.lower())):
            raise ValueError("Required columns: hospital, city, medicine, category, quantity, reorder_level, expiry_date")
        df.columns = df.columns.str.lower()
        if df.empty or len(df) > 5000: raise ValueError("CSV must contain 1–5000 data rows")
        with Session() as db:
            for _, row in df.iterrows():
                hospital = db.scalar(select(Hospital).where(Hospital.name == str(row.hospital)))
                if not hospital:
                    hospital = Hospital(name=str(row.hospital)[:120], city=str(row.city)[:80]); db.add(hospital); db.flush()
                medicine = db.scalar(select(Medicine).where(Medicine.name == str(row.medicine)))
                if not medicine:
                    medicine = Medicine(name=str(row.medicine)[:120], category=str(row.category)[:80]); db.add(medicine); db.flush()
                exp = pd.to_datetime(row.expiry_date, errors="raise").date()
                qty, reorder = int(row.quantity), int(row.reorder_level)
                if qty < 0 or reorder < 0: raise ValueError("Quantity and reorder level must be non-negative")
                db.add(Inventory(hospital_id=hospital.id, medicine_id=medicine.id,
                                 batch_code=f"CSV-{datetime.utcnow():%y%m%d%H%M%S}-{_}", quantity=qty,
                                 reorder_level=reorder, expiry_date=exp))
            db.commit(); audit(db, "CSV_IMPORT", f"Imported {len(df)} inventory rows from {file.filename}")
        return {"imported": len(df)}
    except Exception as exc:
        raise HTTPException(400, f"CSV import failed: {exc}") from exc


@app.post("/api/simulate")
def simulate(req: SimulationRequest):
    with Session() as db:
        item = next((x for x in inventory_payload(db) if x["medicine_id"] == req.medicine_id and x["hospital_id"] == req.hospital_id), None)
        if item is None: raise HTTPException(404, "Inventory combination not found")
        daily = item["daily_forecast"] * (1 + req.demand_change_pct / 100)
        stock = item["quantity"] * (1 + req.supply_change_pct / 100)
        need = max(0, int(round(daily * req.horizon_days)))
        projected = max(0, int(round(need - stock)))
        audit(db, "WHAT_IF_SIMULATION", f"{item['medicine']} at {item['hospital']}: demand {req.demand_change_pct:+g}%, supply {req.supply_change_pct:+g}%, {req.horizon_days} days")
        return {"medicine": item["medicine"], "hospital": item["hospital"], "horizon_days": req.horizon_days,
                "projected_demand": need, "adjusted_stock": int(stock), "projected_gap": projected,
                "status": "Shortage likely" if projected > 0 else "Stock covers scenario",
                "note": "What-if estimate using baseline daily forecast; scenario only, no inventory changed."}
